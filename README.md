# RBCxImperial

### Repo's structure

- `classes`: this folder contains the classes used by the program
- `configs`: configurations files for the API request
- `data`: unzip the content data.zip file into 
- `local`: uploading html page
- `topic_model`: topic classifier pretrained model
- `fake_news_model`: fake news classifier pretrained model
- `main.py`: main python file to launch for both training and inference
- `requirements.txt`: required packages to run the program 
- `tokenizer`: tokenizer embeddings for both fake news and topic classifier

### Requirements

This projects requires python3. The packages used can be found the requirements.txt file and installed by running:

`pip3 install -r requirements.txt`

### Refitting the model

All the models in the project are already trained before uploading. This was done to save time during the testing and demo phase. If refitting the models is necessery run:

`python main.py 1`

The training process takes almost 2 hours. 

### Running the microservice

In order to start the server and loading the models run:

`python main.py`

Once the microservice is online, it will open automatically a web page, at specified host and port,
where you can upload the `dataset_sample.csv` file, located in the `data` folder. 

The upload will return an HTML table with the results. 



## Must read

In order to refit the model you need to extract the content of the following file in the main folder. 
- `mallet-2.0.8.zip`
- `glove.zip`







