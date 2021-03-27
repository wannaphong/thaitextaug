# -*- coding: utf-8 -*-
from typing import List
import gensim.models.keyedvectors as word2vec
import random


class Word2VecAug:
    def __init__(self, model: str, tokenize: object, type = "file"):
        self.tokenizer = tokenize
        if type == "file":
            self.model = word2vec.KeyedVectors.load_word2vec_format(model)
        elif type=="binary":
            self.model = word2vec.KeyedVectors.load_word2vec_format(model, binary=True)
        else:
            self.model = model
    def modify_sent(self,sent, p = 0.7):
        list_sent_new = []
        dict_wv = list(self.model.wv.key_to_index)
        for i in sent:
            if i in dict_wv:
                w = [j for j,v in self.model.most_similar(i) if v>=p]
                if w!=[]:
                    list_sent_new.append(random.choice(w))
                else:
                    list_sent_new.append(i)
            else:
                list_sent_new.append(i)
        return list_sent_new
    def augment(self, sentence: str, n_sent: int = 1, p:int = 0.7) -> List[str]:
        self.sentence = self.tokenizer(sentence)
        self.temp = []
        for i in range(n_sent):
            self.temp += [self.modify_sent(self.sentence, p = p)]
        return self.temp