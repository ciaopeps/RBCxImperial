import pickle
import pandas as pd
import gensim
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow as tf
import gensim.corpora as corpora


class TopicModel(object):

    def __init__(self):
        self.mallet_path = '/Users/pietroaluffi/PycharmProjects/RBCxImperial/mallet-2.0.8/bin/mallet'

        self.df = pd.read_csv('data/dataset.csv')

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

    def lda_df(self):
        lda_data = self.df[self.df['real'] == 1]
        texts = lda_data.clean.astype(str)
        self.texts = [list(text.split(" ")) for text in texts]
        self.id2word = corpora.Dictionary(self.texts)
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]
        print('Log:: LDA database created')
        return self

    def lda_model(self):
        self.model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=self.corpus, num_topics=8,
                                                      id2word=self.id2word)

    @staticmethod
    def extract_topic(ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return (sent_topics_df)

    def transform(self, dataset):
        T = 500
        self.textstest = dataset.clean.astype(str)
        self.sequencestest = self.tokenizer.texts_to_sequences(self.textstest)
        self.datatest = pad_sequences(self.sequencestest, maxlen=T)
        return self

    def get_topic_df(self):
        df_topic = self.extract_topic(self.model, self.corpus, self.texts)
        self.df_topic = df_topic.reset_index()
        self.df_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
        self.df_topic.to_csv('data/topic.csv')
        print('Log:: topic databes saved')
        return self

    def get_labels(self):
        targets = self.df_topic['Dominant_Topic'].values.astype(str)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = targets.reshape(len(targets), 1)
        self.labels = onehot_encoder.fit_transform(integer_encoded)
        self.possible_labels = self.df_topic['Dominant_Topic'].unique()
        print('Log:: target labels obtained')
        return self

    def tokenizer(self):
        # Set Hyperparameters
        V = 20000
        self.texts = self.df_topic.Text.astype(str)

        # Create the Tokenizer
        self.tokenizer = Tokenizer(num_words=V)
        # Fit the tokenizer on the texts to create the word_index dictionary
        self.tokenizer.fit_on_texts(self.texts)
        self.word_index = self.tokenizer.word_index
        # From texts to sequences of integers
        self.sequences = self.tokenizer.texts_to_sequences(self.texts)
        # Set hyperparameters
        T = 500
        # Pad sequences
        self.data = pad_sequences(self.sequences, maxlen=T)
        return self

    def build_model(self):
        print('log:: start building model')
        D = 100
        embedding = Embedding(20000, D)
        bidirectional = Bidirectional(LSTM(D))
        dense = Dense(D, activation='relu')
        output = Dense(8, activation='softmax')
        model = Sequential([
            embedding,
            bidirectional,
            dense,
            output]
        )

        # checkpoint_filepath = data_dir = os.path.join("./gdrive/My Drive/Colab Notebooks/Programming_Session_8/",
        #                                               "model")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def save_all(self):
        """
        Function for saving on disk the pipeline and the model, both fitted.

        :return: self
        """

        with open('tokenizer/tokenizer_lda.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Log:: topic tokenizer saved')

        self.model.save('topic_model')
        print('Log:: topic model saved')

        return self

    def fit(self):
        self.get_labels()
        self.tokenizer()

        """
        Function for fitting the model.

        :param dataset: engeneered dataset
        :return: self
        """
        self.model = self.build_model()

        print('log:: Model built successfully')

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
        with open('tokenizer/tokenizer_lda.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        print('Log:: topic tokenizer loaded')

        self.model = load_model('topic_model')
        print('Log:: topic model loaded')
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
        topics = np.array(['World Politics', 'Middle East', 'Other', 'Election Ivestigation', 'Elections','Europe','US Politics','US Protests'])
        predictions = topics[np.argmax(predictions, axis=1).reshape(-1)]
        print('log:: Predictions obtained')
        return predictions

    def run(self):
        self.lda_df()
        self.lda_model()
        self.get_topic_df()
        self.fit()
        self.save_all()
