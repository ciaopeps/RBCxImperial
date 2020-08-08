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

This repo works in python 3 and in order to run the scripts you need to install the required python libraries.
You can run the following command:

`pip3 install -r requirements.txt`

### Refitting the model

In order to refit (or fitting for the first time), you need to run the following commands:

`python main.py 1`

After a while, the models folder will contain the updated files of the model.

### Running the microservice

In order to run the main.py file you don't have to specify any keywords. 
If you need to change the endpoint and port of the model, you can find their setting
in the `configs` folder. Example of running the microservice:

`python main.py`

Once the microservice is up, it will open automatically a web page, at specified host and port,
where you can upload the `dataset_sample.csv` file, located in the `data` folder.

Once the prediction pipeline has finished it's job, you will be able to see a link on the web page
where you can download the `Predictions.csv` file. 

## Must read

In order to make the pipeline working, you have to save the `dataset_sample.csv` in the `data`
folder with the following columns (order not important):





In case the categorical features don't match the accepted values, their values
must be replaced with the most similar ones provided above.



