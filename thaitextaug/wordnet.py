# -*- coding: utf-8 -*-
"""
Thank https://dev.to/ton_ami/text-data-augmentation-synonym-replacement-4h8l
"""
__all__ = [
    "WordNetAug",
    "postype2wordnet",
]

from pythainlp.corpus import wordnet
from collections import OrderedDict
from pythainlp.tokenize import word_tokenize
from pythainlp.tag import pos_tag
from typing import List
from nltk.corpus import wordnet as wn
import random

lst20 = {
    "": "",
    "AJ": wn.ADJ,
    "AV": wn.ADV,
    "AX": "",
    "CC": "",
    "CL": wn.NOUN,
    "FX": wn.NOUN,
    "IJ": "",
    "NN": wn.NOUN,
    "NU": "",
    "PA": "",
    "PR": "",
    "PS": "",
    "PU": "",
    "VV": wn.VERB,
    "XX": "",
}

orchid = {
    "": "",
    # NOUN
    "NOUN": wn.NOUN,
    "NCMN": wn.NOUN,
    "NTTL": wn.NOUN,
    "CNIT": wn.NOUN,
    "CLTV": wn.NOUN,
    "CMTR": wn.NOUN,
    "CFQC": wn.NOUN,
    "CVBL": wn.NOUN,
    # VERB
    "VACT": wn.VERB,
    "VSTA": wn.VERB,
    # PROPN
    "PROPN": "",
    "NPRP": "",
    # ADJ
    "ADJ": wn.ADJ,
    "NONM": wn.ADJ,
    "VATT": wn.ADJ,
    "DONM": wn.ADJ,
    # ADV
    "ADV": wn.ADV,
    "ADVN": wn.ADV,
    "ADVI": wn.ADV,
    "ADVP": wn.ADV,
    "ADVS": wn.ADV,
    # INT
    "INT": "",
    # PRON
    "PRON": "",
    "PPRS": "",
    "PDMN": "",
    "PNTR": "",
    # DET
    "DET": "",
    "DDAN": "",
    "DDAC": "",
    "DDBQ": "",
    "DDAQ": "",
    "DIAC": "",
    "DIBQ": "",
    "DIAQ": "",
    # NUM
    "NUM": "",
    "NCNM": "",
    "NLBL": "",
    "DCNM": "",
    # AUX
    "AUX": "",
    "XVBM": "",
    "XVAM": "",
    "XVMM": "",
    "XVBB": "",
    "XVAE": "",
    # ADP
    "ADP": "",
    "RPRE": "",
    # CCONJ
    "CCONJ": "",
    "JCRG": "",
    # SCONJ
    "SCONJ": "",
    "PREL": "",
    "JSBR": "",
    "JCMP": "",
    # PART
    "PART": "",
    "FIXN": "",
    "FIXV": "",
    "EAFF": "",
    "EITT": "",
    "AITT": "",
    "NEG": "",
    # PUNCT
    "PUNCT": "",
    "PUNC": "",
}

def postype2wordnet(pos: str, corpus: str):
    if corpus not in ['lst20', 'orchid']:
        return None
    if corpus == 'lst20':
        return lst20[pos]
    else:
        return orchid[pos]



class WordNetAug:
    def __init__(self):
        pass
    def find_synonyms(self, word: str, pos: str = None, postag_corpus: str = "lst20") -> List[str]:
        """
        Find synonyms from wordnet

        :param str word: word
        :param str pos: part of speech
        :param str postag_corpus: postag corpus name
        :return: list of synonyms
        """
        self.synonyms = []
        if pos == None:
            self.list_synsets = wordnet.synsets(word)
        else:
            self.p2w_pos = postype2wordnet(pos, postag_corpus)
            if self.p2w_pos != '':
                self.list_synsets = wordnet.synsets(word, pos=self.p2w_pos)
            else:
                self.list_synsets = wordnet.synsets(word)

        for self.synset in wordnet.synsets(word):
            for self.syn in self.synset.lemma_names(lang='tha'):
                self.synonyms.append(self.syn)

        # using this to drop duplicates while maintaining word order (closest synonyms comes first)
        self.synonyms_without_duplicates = list(OrderedDict.fromkeys(self.synonyms))
        return self.synonyms_without_duplicates
    def gen_sent(self,list_words: list,dict_synonym: dict,index: int):
        """
        :param list list_words: list words
        :param dict dict_synonym: dict synonym words
        :param int index: index of list words

        :return: list of sent
        """
        word = list_words[index]
        w = random.choice(dict_synonym[word])
        if index!=len(list_words)-1:
            index+=1
            return [w]+self.gen_sent(list_words,dict_synonym,index)
        else:
            return [w]
    def augment(self, sentence: str, tokenize: object = word_tokenize, max_syn_sent: int = 6, postag: bool = True, postag_corpus: str = "lst20") -> List[List[str]]:
        """
        Text Augment using wordnet

        :param str sentence: thai sentence
        :param object tokenize: function for tokenize word
        :param int max_syn_sent: number max for synonyms sent
        :param bool postag: on part-of-speech
        :param str postag_corpus: postag corpus name

        :return: list of synonyms
        """
        new_sentences = []
        self.list_words = word_tokenize(sentence)
        self.list_synonym = {w:[] for w in self.list_words}
        self.p_all = 1
        if postag:
            self.list_pos = pos_tag(self.list_words, corpus=postag_corpus)
            for word, pos in self.list_pos:
                self.temp = self.find_synonyms(word, pos, postag_corpus)
                if self.temp == []:
                    self.list_synonym[word] = [word]
                else:
                    self.list_synonym[word] = self.temp
                    self.p_all*= len(self.temp)
        else:
            for word in self.list_words:
                self.list_synonym[word] = self.find_synonyms(word) 
                if self.temp == []:
                    self.list_synonym[word] = [word]
                else:
                    self.list_synonym[word] = self.temp
                    self.p_all*= len(self.temp)
        if max_syn_sent > self.p_all:
            max_syn_sent = self.p_all
        i = 0
        while len(new_sentences)<max_syn_sent:
            self.t = self.gen_sent(self.list_words,self.list_synonym,0)
            if self.t not in new_sentences:
                new_sentences.append(self.t)
                i=0
            else:
                i+=1
        return new_sentences