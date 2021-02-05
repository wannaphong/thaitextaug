# -*- coding: utf-8 -*-
import nlpaug.augmenter.word as naw
from pythainlp.corpus import get_corpus_path
from pythainlp.tokenize import THAI2FIT_TOKENIZER
import re
from typing import List


class Word2Vec:
    def __init__(self, model_path: str, tokenize: object, action: str):
        self.wn_path = model_path
        self.tokenizer = tokenize
        self.aug = naw.WordEmbsAug(
            model_type='word2vec', model_path= self.wn_path,
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


class Thai2fit:
    def __init__(self):
        self.thai2fit_wv = get_corpus_path('thai2fit_wv')
        self.load_w2v()
        self.word_postprocessing = re.compile(r'(\w) (\w)')
    def tokenizer_thai2fit(self, text: str):
        return THAI2FIT_TOKENIZER.word_tokenize(text)
    def load_w2v(self, action="substitute"): # insert substitute
        self.aug = Word2Vec(self.thai2fit_wv, self.tokenizer_thai2fit, action)
    def augment(self, sentence: str, n_sent: int = 1) -> List[str]:
        """
        Text Augment using word2vec from Thai2Fit

        :param str sentence: thai sentence
        :param int n_sent: number sentence

        :return: list of synonyms
        """
        list_aug = [
            self.word_postprocessing.sub(r'\1\2',i).replace('   ',' ') for i in self.aug.augment(sentence, n_sent)
        ]
        return list_aug

from .BPEmb import BPEmb