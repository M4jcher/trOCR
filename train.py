import os
import numpy as np
import pandas as pd
import cv2
import torch
import json
from licensePlates import licensePlates
from PIL import Image
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from evaluate import load
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

output_dir = 'D:\\Pulpit\\trocr\\output'
os.makedirs(output_dir, exist_ok=True)


def get_string(file):
    with open(file, 'r') as f:
        data = f.read()

    # przekazanie danych do parsera
    bs_data = BeautifulSoup(data, 'xml')

    # znajduje wszystkie elementy oznaczone tagiem "obiect"
    b_unique = bs_data.find_all('object')
    return ''.join([i.find('name').text for i in b_unique])


def countCord(data_dict):
    cord = {}

    original_width = data_dict['original_width']
    original_height = data_dict['original_height']

    pixel_x = int(data_dict['x'] / 100. * original_width)
    pixel_y = int(data_dict['y'] / 100. * original_height)
    pixel_width = int(data_dict['width'] / 100. * original_width)
    pixel_height = int(data_dict['height'] / 100. * original_height)
    cord['x1'] = pixel_x - 2
    cord['y1'] = pixel_y - 2
    cord['x2'] = pixel_x + pixel_width + 5
    cord['y2'] = pixel_y + pixel_height + 5
    return cord


def preprocess(image, width: int, height: int, cval: int = 255, mode="letterbox", return_scale=False, ):
    fitted = None
    x_scale = width / image.shape[1]
    y_scale = height / image.shape[0]
    if x_scale == 1 and y_scale == 1:
        fitted = image
        scale = 1
    elif (x_scale <= y_scale and mode == "letterbox") or (
            x_scale >= y_scale and mode == "crop"
    ):
        scale = width / image.shape[1]
        resize_width = width
        resize_height = (width / image.shape[1]) * image.shape[0]
    else:
        scale = height / image.shape[0]
        resize_height = height
        resize_width = scale * image.shape[1]
    if fitted is None:
        resize_width, resize_height = map(int, [resize_width, resize_height])
        if mode == "letterbox":
            fitted = np.zeros((height, width, 3), dtype="uint8") + cval
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted[: image.shape[0], : image.shape[1]] = image[:height, :width]
        elif mode == "crop":
            image = cv2.resize(image, dsize=(resize_width, resize_height))
            fitted = image[:height, :width]
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")
    if not return_scale:
        return fitted
    return fitted, scale


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


def LP_compare(true_label, pred_label):
    true_label = true_label.replace(" ", "").replace('-', '').replace('0', 'O')
    pred_label = pred_label.replace(" ", "").replace('-', '').replace('0', 'O')
    if pred_label == true_label:
        return True

    return False


# image_dir = 'D:\\Pulpit\\trocr\\license_plate_characters\\LP-characters\\images'
# annotations_dir = 'D:\\Pulpit\\trocr\\license_plate_characters\\LP-characters\\annotations'
#
# lP_records2 = []
#
# for im_name in sorted(os.listdir(image_dir)):
#     image_path = os.path.join(image_dir, im_name)
#
#     annot_path = os.path.join(annotations_dir, im_name.split('.')[0] + '.xml')
#     label_string = get_string(annot_path)
#
#     lP_records2.append(dict(text=str(label_string), file_name=image_path))
#
# df1 = pd.DataFrame(lP_records2)
# df1.to_csv('D:\\Pulpit\\trocr\\working\\LP_records2.csv')

labelsData = pd.read_csv('D:\\Pulpit\\trocr\\tags\\ocr-licence-plate.csv')
labelsData['ocr'] = labelsData['ocr'].apply(lambda x: x.split('-')[-1])

LP_records = []

path_car_img = "D:\\Pulpit\\trocr\\images"
os.makedirs("D:\\Pulpit\\trocr\\working\\images\\", exist_ok=True)
for row, val in labelsData.iterrows():
    img_dir = os.path.join(path_car_img, val['ocr'])
    image = cv2.imread(img_dir, cv2.IMREAD_ANYCOLOR)
    try:
        label_s = json.loads(val['transcription'])
    except ValueError:
        label_s = str(val['transcription'])
    for num, bbox in enumerate(json.loads(val['bbox'])):
        path_to_save = f"D:\\Pulpit\\trocr\\working\\images\\{val['ocr'].split('.')[0]}_{num}.jpg"

        cart_cord = countCord(bbox)
        crop_img = image[cart_cord['y1']:cart_cord['y2'],
                   cart_cord['x1']:cart_cord['x2']]
        crop_img = preprocess(crop_img, width=200, height=100)

        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        im_croped = Image.fromarray(crop_img)

        im_croped.save(path_to_save)
        if isinstance(label_s, list):
            LP_records.append(dict(text=str(label_s[num]), file_name=path_to_save))
        else:
            LP_records.append(dict(text=str(label_s), file_name=path_to_save))

df2 = pd.DataFrame(LP_records)
df2.to_csv('D:\\Pulpit\\trocr\\working\\LP_records.csv')
df2 = df2.sort_index()

# test_df = pd.concat([df1[-21:], df2[-21:]])
# df = pd.concat([df1[:-21], df2[:-21]])
test_df = df2.iloc[:-21]
df = df2.iloc[:-21]
# train_df, valid_df = train_test_split(df[:-21], test_size=0.2)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=0)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

##### TROCR #####

processor=TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model=VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

root_dir = ''

train_dataset = licensePlates(root_dir, df=train_df, processor=processor)

test_dataset = licensePlates(root_dir, df=test_df, processor=processor)

eval_dataset = licensePlates(root_dir, df=valid_df, processor=processor)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

model.config.vocab_size = model.config.decoder.vocab_size

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 3

cer_metric = load("cer")

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    overwrite_output_dir=True,
    output_dir=output_dir,
    logging_steps=2,
    save_steps=300,
    eval_steps=100,
    report_to="none",
    num_train_epochs=2,
)

# inicjalizacja trenera
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# trainer.train()
# train_time = time.time() - train_time
#
# print('Train time (sec)', train_time)
#
# save_time = time.time()
# model.save_pretrained("D:\\Pulpit\\trocr\\working\\vit-ocr")
# save_time = time.time() - save_time
# print('save_time time (sec)', save_time)
#
# score_model = VisionEncoderDecoderModel.from_pretrained('D:\\Pulpit\\trocr\\working\\vit-ocr')