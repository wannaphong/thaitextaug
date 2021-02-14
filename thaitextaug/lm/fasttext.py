# -*- coding: utf-8 -*-
from typing import List
from gensim.models.fasttext import FastText
from pythainlp.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import random

class FastTextAug:
    def __init__(self, model_path: str, model_type = "fb"):
        if model_type == "fb" and model_path.endswith('.bin'):
            self.model = FastText.load_fasttext_format(model_path)
        elif model_type == "fb" and model_path.endswith('.vec'):
            self.model = KeyedVectors.load_word2vec_format(model_path)
        else:
            self.model = FastText.load(model_path)
    def tokenize(self, text):
        return word_tokenize(text, engine='icu')
    def modify_sent(self,sent, p = 0.7):
        list_sent_new = []
        dict_wv = list(self.model.vocab.keys())
        for i in sent:
            if i in dict_wv:
                w = [j for j,v in self.model.most_similar(i) if v>=p]
                if w != []:
                    list_sent_new.append(random.choice(w))
                else:
                    list_sent_new.append(i)
            else:
                list_sent_new.append(i)
        return list_sent_new
    def augment(self, sentence: str, n_sent: int = 1, p:int = 0.7) -> List[str]:
        self.sentence = self.tokenize(sentence)
        self.temp = []
        for i in range(n_sent):
            self.temp += [self.modify_sent(self.sentence, p = p)]
        return self.temp