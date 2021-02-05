# -*- coding: utf-8 -*-
import nlpaug.augmenter.word as naw
from typing import List
import gensim.models.keyedvectors as word2vec
import random

class NLPaug:
    def __init__(self, model_path: str, tokenize: object, action: str, model_type: str = 'word2vec'):
        self.wn_path = model_path
        self.tokenizer = tokenize
        self.aug = naw.WordEmbsAug(
            model_type=model_type, model_path= self.wn_path,
            action=action,tokenizer=self.tokenizer
        )
    def augment(self, sentence: str, n_sent: int = 1) -> List[str]:
        """
        Text Augment using word2vec

        :param str sentence: thai sentence
        :param int n_sent: number sentence

        :return: list of synonyms
        """
        return self.aug.augment(sentence, n=n_sent)

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
        dict_wv = list(self.model.vocab.keys())
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
        self.temp = []
        for i in range(n_sent):
            self.temp += [self.modify_sent(sentence, p = p)]
        return self.temp