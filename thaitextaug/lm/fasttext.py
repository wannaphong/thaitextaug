# -*- coding: utf-8 -*-
from typing import List, Tuple
from gensim.models.fasttext import FastText as FastText_gensim
from pythainlp.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import random

class FastTextAug:
    """
    Text Augment from FastText
    """
    def __init__(self, model_path: str):
        """
        :param str model_path: path of model file
        """
        if model_path.endswith('.bin'):
            self.model = FastText_gensim.load_facebook_vectors(model_path)
        elif model_path.endswith('.vec'):
            self.model = KeyedVectors.load_word2vec_format(model_path)
        else:
            self.model = FastText_gensim.load(model_path)
    def tokenize(self, text: str)-> List[str]:
        """
        Thai text tokenize for fasttext

        :param str text: thai text

        :return: list of word
        :rtype: List[str]
        """
        return word_tokenize(text, engine='icu')
    def modify_sent(self,sent: list, p: float = 0.7) -> Tuple[str]:
        list_sent_new = []
        dict_wv = list(self.model.key_to_index.keys())
        for i in sent:
            if i in dict_wv:
                w = [j for j,v in self.model.most_similar(i) if v>=p]
                if w != []:
                    list_sent_new.append(random.choice(w))
                else:
                    list_sent_new.append(i)
            else:
                list_sent_new.append(i)
        return tuple(list_sent_new)
    def augment(self, sentence: str, n_sent: int = 1, p:float = 0.7) -> List[Tuple[str]]:
        """
        Text Augment from FastText

        You wants to download thai model from https://fasttext.cc/docs/en/crawl-vectors.html.

        :param str sentence: thai sentence
        :param int n_sent: number sentence
        :param float p: Probability of word

        :return: list of synonyms
        :rtype: List[Tuple[str]]
        """
        self.sentence = self.tokenize(sentence)
        self.temp = []
        for i in range(n_sent):
            self.temp += [self.modify_sent(self.sentence, p = p)]
        return self.temp