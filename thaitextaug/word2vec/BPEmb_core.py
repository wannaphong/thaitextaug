# -*- coding: utf-8 -*-
from thaitextaug.word2vec import Word2VecAug
from bpemb import BPEmb
from typing import List

class BPEmbAug:
    def __init__(self, lang: str = "th", vs:int = 100000, dim: int = 300):
        self.bpemb_temp = BPEmb(lang=lang, dim=dim, vs= vs)
        self.model = self.bpemb_temp.emb
        self.load_w2v()
    def tokenizer(self, text: str):
        return self.bpemb_temp.encode(text)
    def load_w2v(self):
        self.aug = Word2VecAug(self.model, tokenize=self.tokenizer, type="model")
    def augment(self, sentence: str, n_sent: int = 1, p = 0.7) -> List[str]:
        self.sentence =  sentence.replace(" ","▁")
        self.temp = self.aug.augment(self.sentence, n_sent, p = p)
        self.temp_new = []
        for i in self.temp:
            self.t = ""
            for j in i:
                self.t += j.replace('▁','')
            self.temp_new.append(self.t)
        return self.temp_new