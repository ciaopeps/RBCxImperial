import pandas as pd
import os
import urllib
import gensim
from spacy.lang.en.stop_words import STOP_WORDS as en
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy #python3 -m spacy download it_core_news_sm


class Preprocessing(object):

    def __init__(self):
        pass

    @staticmethod
    def mining(real):
        # removing publisher
        texts = real.text
        articles = []
        for text in texts:
            try:
                articles.append(text.split('-', maxsplit=1)[1])
            except:
                articles.append([text])
        real['clean'] = articles
        stop_words = list(en)
        url = "https://gist.githubusercontent.com/deekayen/4148741/raw/98d35708fa344717d8eee15d11987de6c8e26d7d/1-1000.txt"
        file = urllib.request.urlopen(url)
        stop_ext = [line.decode("utf-8").strip() for line in file]
        stop_words = stop_words + stop_ext
        stop_words.extend(
            ['use', 'per', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try',
             'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want',
             'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
        lemma = spacy.load('en_core_web_sm', disable=['parser', 'Ner'])

        def token(sentences):
            for sentence in sentences:
                yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

        def bigram_trigram(t):
            bigram = gensim.models.Phrases(t)
            trigram = gensim.models.Phrases(bigram[t])
            bigram_ = gensim.models.phrases.Phraser(bigram)
            trigram_ = gensim.models.phrases.Phraser(trigram)
            texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in t]
            bi = [bigram_[text] for text in texts]
            tri = [trigram_[bigram_[b]] for b in bi]
            return tri

        def processing(texts, stop_words=stop_words):
            texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            output = []
            for text in texts:
                doc = lemma(" ".join(text))
                output.append([token.lemma_ for token in doc])
            # remove stopwords once more after lemmatization
            output = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in output]
            return output

        tokens = list(token(articles))
        bi_tri = bigram_trigram(tokens)
        final_data = processing(bi_tri)
        real['clean'] = [' '.join(final) for final in final_data]
        return real


    def extract(self):
        """
        Function for generating the datasets

        :return: tuple of extracted and mined dataset
        """

        print('log:: Start loading files...')

        real = pd.read_csv(os.path.join('data', 'True.csv'))
        fake = pd.read_csv(os.path.join('data', 'Fake.csv'))
        print('log:: Files loaded')

        real = self.mining(real)
        fake = self.mining(fake)
        real['real'] = 1
        fake['fake'] = 1
        dataset = pd.concat([real, fake]).fillna(0)

        print('log:: Data mined')

        dataset.sample(2500).to_csv(os.path.join('data', 'dataset_sample.csv'), encoding='utf8', sep=',', index=True)
        dataset.to_csv(os.path.join('data', 'dataset.csv'), encoding='utf8', sep=',', index=True)
        print('log:: Dataset exported')

        return dataset.reset_index(drop=False), dataset.sample(2500).reset_index(drop=False)
