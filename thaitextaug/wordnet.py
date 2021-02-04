# -*- coding: utf-8 -*-
"""
Thank https://dev.to/ton_ami/text-data-augmentation-synonym-replacement-4h8l
"""
from pythainlp.corpus import wordnet
from collections import OrderedDict
from pythainlp.tokenize import word_tokenize
from typing import List


class WordNet:
    def __init__(self):
        pass
    def find_synonyms(self, word: str) -> List[str]:
        """
        Find synonyms from wordnet

        :param str word: word
        :return: list of synonyms
        """
        self.synonyms = []
        for self.synset in wordnet.synsets(word):
            for self.syn in self.synset.lemma_names(lang='tha'):
                self.synonyms.append(self.syn)

        # using this to drop duplicates while maintaining word order (closest synonyms comes first)
        self.synonyms_without_duplicates = list(OrderedDict.fromkeys(self.synonyms))
        return self.synonyms_without_duplicates
    def augment(self, sentence: str, tokenize: object = word_tokenize, max_syn_per_word: int = 6) -> List[str]:
        """
        Text Augment using wordnet

        :param str sentence: thai sentence
        """
        new_sentences = []
        for word in word_tokenize(sentence):
            if len(word)<=3 : continue 
            for synonym in self.find_synonyms(word)[0:max_syn_per_word]:
                synonym = synonym.replace('_', ' ') #restore space character
                new_sentence = sentence.replace(word,synonym,1)
                new_sentences.append(new_sentence)
        return new_sentences