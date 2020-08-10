# RBCxImperial

### Repo's structure

- `classes`: this folder contains the classes used by the program
- `configs`: configurations files for the API request
- `data`: CSV files witht the dataset. True.csv and Fake.csv are the original datasets. dataset.csv and sample_dataset.csv are obtained after preprocessing. Use sample_dataset.csv to test the system 
- `local`: uploading html page
- `topic_model`: topic classifier pretrained model
- `fake_news_model`: fake news classifier pretrained model
- `main.py`: main python file to launch for both training and inference
- `requirements.txt`: required packages to run the program 
- `tokenizer`: tokenizer embeddings for both fake news and topic classifier
- `imperialXrbc.ipynb`: Google Colab notebook with the project Proof of Concept

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



# Must read
Make sure to run and install all requirements and the following command:

`python3 -m spacy download it_core_news_lg `

Big files such as _glove.txt_, _Real.csv_, _Fake.csv_ and _dataset.csv_ need to be extracted after downloading the repo. 






