# -*- coding: utf-8 -*-
from typing import List
import gensim.models.keyedvectors as word2vec
import random


class Word2VecAug:
    def __init__(self, model: str, tokenize: object, type: str = "file") -> None:
        """
        :param str model: path model
        :param object tokenize: tokenize function
        :param str type: moodel type (file, binary)
        """
        self.tokenizer = tokenize
        if type == "file":
            self.model = word2vec.KeyedVectors.load_word2vec_format(model)
        elif type=="binary":
            self.model = word2vec.KeyedVectors.load_word2vec_format(model, binary=True)
        else:
            self.model = model
        self.dict_wv = list(self.model.vocab.keys())
    def modify_sent(self,sent, p = 0.7):
        """
        :param str sent: text sentence
        :param int p: probability
        :rtype: List[str]
        """
        list_sent_new = []
        for i in sent:
            if i in self.dict_wv:
                w = [j for j,v in self.model.most_similar(i) if v>=p]
                if w!=[]:
                    list_sent_new.append(random.choice(w))
                else:
                    list_sent_new.append(i)
            else:
                list_sent_new.append(i)
        return list_sent_new
    def augment(self, sentence: str, n_sent: int = 1, p:int = 0.7) -> List[List[str]]:
        """
        :param str sentence: text sentence
        :param int n_sent: max number for synonyms sentence
        :param int p: probability

        :return: list of synonyms
        :rtype: List[List[str]]
        """
        self.sentence = self.tokenizer(sentence)
        self.temp = []
        for i in range(n_sent):
            self.temp += [self.modify_sent(self.sentence, p = p)]
        return self.temp