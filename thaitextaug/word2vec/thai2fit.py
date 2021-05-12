# -*- coding: utf-8 -*-
from thaitextaug.word2vec import Word2VecAug
from pythainlp.corpus import get_corpus_path
from pythainlp.tokenize import THAI2FIT_TOKENIZER
from typing import List, Tuple


class Thai2fitAug:
    """
    Text Augment using word2vec from Thai2Fit

    Thai2Fit: `github.com/cstorm125/thai2fit <https://github.com/cstorm125/thai2fit>`_
    """
    def __init__(self):
        self.thai2fit_wv = get_corpus_path('thai2fit_wv')
        self.load_w2v()
    def tokenizer(self, text: str) -> List[str]:
        """
        :param str text: thai text
        :rtype: List[str]
        """
        return THAI2FIT_TOKENIZER.word_tokenize(text)
    def load_w2v(self): # insert substitute
        """
        Load thai2fit word2vec model
        """
        self.aug = Word2VecAug(self.thai2fit_wv, self.tokenizer, type="binary")
    def augment(self, sentence: str, n_sent: int = 1, p: int = 0.7) -> List[Tuple[str]]:
        """
        Text Augment using word2vec from Thai2Fit

        :param str sentence: thai sentence
        :param int n_sent: number sentence
        :param float p: Probability of word

        :return: list of synonyms
        :rtype: List[Tuple[str]]
        """
        return self.aug.augment(sentence, n_sent, p)