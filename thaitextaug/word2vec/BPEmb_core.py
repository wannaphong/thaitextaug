from thaitextaug.word2vec import Word2VecAug
from bpemb import BPEmb
from typing import List

class BPEmb_aug:
    def __init__(self, lang: str = "th", dim: int = 30):
        self.bpemb_temp = BPEmb(lang=lang, dim=dim)
        self.model = self.bpemb_temp.emb
        self.load_w2v()
    def tokenizer(self, text):
        return self.bpemb_temp.encode(text)
    def load_w2v(self, action="substitute"):
        self.aug = Word2VecAug(self.model, self.tokenizer, type="model")
    def augment(self, sentence: str, n_sent: int = 1, p = 0.7) -> List[str]:
        self.sentence =  sentence.replace(" ","▁")
        self.word_list = self.tokenizer(self.sentence)
        self.temp = self.aug.augment(self.word_list, n_sent, p = p)
        self.temp_new = []
        for i in self.temp:
            self.t = ""
            for j in i:
                self.t += j.replace('▁','')
            self.temp_new.append(self.t)
        return self.temp_new