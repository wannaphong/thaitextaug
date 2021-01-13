# -*- coding: utf-8 -*-
import nlpaug.augmenter.word as naw
from pythainlp.corpus import get_corpus_path
from pythainlp.tokenize import THAI2FIT_TOKENIZER
import re


class Thai2fit:
    def __init__(self):
        self.thai2fit_wv = get_corpus_path('thai2fit_wv')
        self.load_w2v()
        self.word_postprocessing = re.compile(r'(\w) (\w)')
    def tokenizer_thai2fit(self, text):
        return THAI2FIT_TOKENIZER.word_tokenize(text)
    def load_w2v(self, action="substitute"):
        self.aug = naw.WordEmbsAug(
            model_type='word2vec', model_path=self.thai2fit_wv,
            action=action,tokenizer=self.tokenizer_thai2fit
        )
    def augment(self, sentence: str, n_sent: int = 6):
        list_aug = [
            self.word_postprocessing.sub(r'\1\2',i).replace('   ',' ') for i in self.aug.augment(sentence,n=n_sent)
        ]
        return list_aug