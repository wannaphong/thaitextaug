# -*- coding: utf-8 -*-
import unittest
from thaitextaug.wordnet import WordNet


class WordNetPackage(unittest.TestCase):
    def test_WordNet(self):
        w = WordNet()
        text = "ประกาศมหาวิทยาลัย ฉบับที่ 6"
        self.assertIsNotNone(w.augment(text))
