from .word2vec import Word2Vec
from bpemb import BPEmb

class BPEmb:
    def __init__(self, lang: str = "th", dim: int = 30):
        self.bpemb_temp = BPEmb(lang=lang, dim=dim)
        self.model_path = self.bpemb_temp.emb_file
    def tokenizer(self, text):
        return self.bpemb_temp.encode(text)
    def load_w2v(self, action="substitute"):
        self.aug = Word2Vec(self.model_path, self.tokenizer, action)
    def augment(self, sentence: str, n_sent: int = 1) -> List[str]:
        return self.aug.augment(sentence, n_sent)