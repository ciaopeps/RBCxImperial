import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding
import os


class ClassifierModel(object):

    def __init__(self):

        self.V = 20000
        self.T = 500
        self.D = 100
        self.callback_list = [
            EarlyStopping(  # Interrupt the training when there is no more improvement
                monitor="val_accuracy",  # Metric to monitor
                patience=2  # Number of epochs before interrupting the training if there is no improvement
            ),
            # ModelCheckpoint(
            #     filepath = checkpoint_filepath, # path to the destination model file
            #     monitor = "val_loss", # metric to monitor
            #     save_best_only = True # Keep the best model
            # ),
            ReduceLROnPlateau(
                monitor="val_loss",  # metric to monitor
                factor=0.1,  # Multiply the lr by factor when triggered
                patience=2  # Number of epochs of non improvement before the callback is triggered
            )]

    def get_labels(self, dataset):
        # List of possible labels
        possible_labels = ['real', 'fake']
        # Matrix of labels of shape (N, K)
        labels = dataset[possible_labels].values
        self.labels = np.asarray(labels)
        print('labels done')
        return self

    def tokenizer(self, dataset):
        # Set Hyperparameters
        V = 20000
        self.texts = dataset.clean.astype(str)
        # Create the Tokenizer
        self.tokenizer = Tokenizer(num_words=V)
        # Fit the tokenizer on the texts to create the word_index dictionary
        self.tokenizer.fit_on_texts(self.texts)
        # with open('tokenizer/tokenizer.pickle', 'wb') as handle:
        #     pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('tokenizer saved')
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))

        # From texts to sequences of integers
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        # Set hyperparameters
        T = 500
        # Pad sequences
        self.data = pad_sequences(self.sequences, maxlen=T)
        print('token done')
        return self

    def transform(self, dataset):
        T = 500
        self.textstest = dataset.clean.astype(str)
        self.sequencestest = self.tokenizer.texts_to_sequences(self.textstest)
        self.datatest = pad_sequences(self.sequencestest, maxlen=self.T)
        return self

    def embedding_layer(self):
        # Directory of the Embedding file
        dir_glove = os.path.join('glove/')

        # Create the dictionary
        embedding_index = {}

        f = open(os.path.join(dir_glove, "glove.6B.100d.txt"), encoding="utf8")
        for line in f:
            # Split each line
            values = line.split()
            # Get the word
            word = values[0]
            # Get the embedding vector
            embedding_vector = np.asarray(values[1:], dtype="float32")
            # Append the dictionary
            embedding_index[word] = embedding_vector
        f.close()
        # The embedding dimension

        # Initilize the embedding matrix with zeros
        embedding_matrix = np.zeros((self.V, self.D))

        # Loop through all the elements of the word_index dictionary
        for word, i in self.word_index.items():
            if i < self.V:
                # Get the embedding vector or None
                embedding_vector = embedding_index.get(word)
                # Update one row of the embedding matrix
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(
            self.V,
            self.D,
            weights=[embedding_matrix],
            input_length=self.T,
            trainable=False
        )
        print('Log:: embedding layer created')
        return self

    def build_model(self):
        possible_labels = [0, 1]
        # Set hyperparameters
        d_1 = 32  # output dimension of the first LSTM layer
        d_2 = 16  # output dimension of the second LSTM layer
        d_3 = len(possible_labels)  # Output dimension of the last layer

        # Define the two lstm layers
        lstm_1 = LSTM(d_1, return_sequences=True)
        lstm_2 = LSTM(d_2, return_sequences=False)

        # The input layer
        input_layer = Input(shape=(self.T,), dtype="int32")  # (N, T)

        # Apply the embeddin lyer
        x = self.embedding_layer(input_layer)  # (N, T, D)

        # Apply the first LSTM layer
        y = lstm_1(x)  # (N, T, d_1)

        # Apply the second LSTM layer
        h_T = lstm_2(y)  # (N, d_2)

        # Apply the dense layer
        output = Dense(d_3, activation="sigmoid")(h_T)  # (N, d_3)

        # Define the model
        model = Model(input_layer, output)
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        print('Log:: model built')
        return model

    def save_all(self):
        """
        Function for saving on disk the pipeline and the model, both fitted.

        :return: self
        """

        with open('tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Log:: tokenizer saved')

        self.model.save('fake_news_model')
        print('Log:: model_saved')

        return self

    def fit(self, dataset):
        self.get_labels(dataset)
        self.tokenizer(dataset)
        # self.shuffle()
        self.embedding_layer()

        """
        Function for fitting the model.

        :param dataset: engeneered dataset
        :return: self
        """
        self.model = self.build_model()



        self.model.fit(
            self.data, self.labels,
            batch_size=32, shuffle=True, epochs=20, verbose=True,
            validation_split=.2,
            callbacks=self.callback_list
        )
        print('log:: Model fitted successfully')

        self.save_all()

        return self

    def load_all(self):
        """
        Function for loading the model and the pipeline.

        :return: self
        """
        with open('tokenizer/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        print('Log:: tokenizer loaded')
        self.model = load_model('fake_news_model')
        print('Log:: model loaded')
        return self

    def serve_predictions(self, new_X):
        """
        Function for making the prediction on new data

        :param new_X: new observations
        :return: predictions
        """
        self.load_all()
        self.transform(new_X)
        predictions = self.model.predict(self.datatest)
        predictions = np.argmax(predictions, axis=1).reshape(-1)

        print('log:: Predictions obtained for Fake/Real news')
        return predictions
