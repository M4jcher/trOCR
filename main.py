from train import processor
from train import model
from train import test_df
from train import LP_compare
from train import df2
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from PIL import Image


def main():
    correct_char = 0
    correct = 0
    avg = 0

    first_df = df2.iloc[:101]
    second_df = df2.iloc[101:201]
    third_df = df2.iloc[201:301]
    rest_df = df2.iloc[-39:]

    # print(first_df)
    # print(second_df)
    # print(third_df)
    # print(rest_df)

    # set dataset to test
    df = rest_df

    full_time = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for _, val in test_df.iterrows():
        rec_time = time.time()

        # rozpoznawanie - obraz przetwarzany na tekst
        image = Image.open(val['file_name']).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        # pixel_values = pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        pred_label = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        true_label = val['text']

        correct_char += SequenceMatcher(None, true_label, pred_label).ratio()

        rec_time = time.time() - rec_time
        avg += rec_time
        if LP_compare(true_label, pred_label):
            correct += 1
        else:
            # plt.imshow(np.asarray(image))
            # plt.show()
            print('true_label', true_label)
            print('pred_label', pred_label)
            print('Recognition time: ', rec_time)

        # czyszczenie pamięci GPU
        # torch.cuda.empty_cache()

    full_time = time.time() - full_time

    print(f'correct {correct}, total {len(df)}')
    print('Correct characters predicted: %.2f%%' % (correct_char * 100 / df.shape[0]))
    print('Correct license plates predicted: %.2f%%' % (correct * 100 / df.shape[0]))
    print('Avarage recognition time: ', avg / len(df))
    print('Full process time: ', full_time)


if __name__ == '__main__':
    main()