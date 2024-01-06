<h1> RaspPlate recognition by transformer </h1>
<h2> Wojciech Majchrzak 229947 </h2>
<h3> 🛠 About project </h3>

- &nbsp; The project uses transformer technic to extract text from photos
- &nbsp; To use it the user needs to open terminal and provide: **pip install transformers**
- &nbsp; In the project are used many different libraries to help analyze data like pandas, Image, BeautifulSoup, OpenCV
- &nbsp; We can modify training arguments to create our transformer engine
- &nbsp; The path to the engine is: **working/vit-ocr**
- &nbsp; The engine is alredy created and there is no need to train a model again but the user can recreate and overwrite it when it is needed
- &nbsp; **train.py** class contains all functions responsible for preprocessing images, prepare dataframes and recognition model
- &nbsp; **main.py** class consists of loop going through all pictures and converting to the values which are passed to the decoder model which recognize the text
- &nbsp; **licensePlate.py** class is used to load data during the process of learning the model (create recognition engine) 
