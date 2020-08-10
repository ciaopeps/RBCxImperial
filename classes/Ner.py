from collections import Counter
import en_core_web_lg
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

nlp = en_core_web_lg.load()


class Ner(object):
    @staticmethod
    def get_people(text):
        doc = nlp(text)
        ppl = [(ent.text) for ent in doc.ents if ent.label_ == 'PERSON']
        c = Counter(ppl)
        ppl = [c.most_common(3)[i][0] for i in range(min(len(c),3))]
        return ppl

    @staticmethod
    def get_org(text):
        doc = nlp(text)
        org = [(ent.text) for ent in doc.ents if ent.label_ == 'ORG']
        c = Counter(org)
        org = [c.most_common(3)[i][0] for i in range(min(len(c),3))]
        return org

    @staticmethod
    def summary(text):
        model = T5ForConditionalGeneration.from_pretrained('t5-small')

        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')
        preprocess_text = text.strip().replace("\n", "")
        t5_prepared_Text = "summarize: " + preprocess_text
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        # summmarize
        summary_ids = model.generate(tokenized_text,
                                     num_beams=4,
                                     no_repeat_ngram_size=2,
                                     min_length=30,
                                     max_length=800,
                                     early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output

    def run(self, dataset):
        """
        Function for running the class

        :param dataset: input dataset
        :return: engeneered dataset
        """

        dataset['ppl'] = dataset['text'].apply(self.get_people)
        dataset['org'] = dataset['text'].apply(self.get_org)

        return dataset
