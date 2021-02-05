# -*- coding: utf-8 -*-
from thaitextaug.word2vec import Word2VecAug
import nlpaug.augmenter.word as naw
from pythainlp.corpus import get_corpus_path
from pythainlp.tokenize import THAI2FIT_TOKENIZER
from typing import List


class Thai2fit:
    def __init__(self):
        self.thai2fit_wv = get_corpus_path('thai2fit_wv')
        self.load_w2v()
    def tokenizer_thai2fit(self, text: str):
        return THAI2FIT_TOKENIZER.word_tokenize(text)
    def load_w2v(self): # insert substitute
        self.aug = Word2VecAug(self.thai2fit_wv, self.tokenizer_thai2fit, type="binary")
    def augment(self, sentence: str, n_sent: int = 1) -> List[str]:
        """
        Text Augment using word2vec from Thai2Fit

        :param str sentence: thai sentence
        :param int n_sent: number sentence

        :return: list of synonyms
        """
        return self.aug.augment(sentence, n_sent)