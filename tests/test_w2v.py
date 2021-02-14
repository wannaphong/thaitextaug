# -*- coding: utf-8 -*-
import unittest
from thaitextaug.word2vec import Thai2fitAug


class Word2VecPackage(unittest.TestCase):
    def test_thai2fit(self):
        w2v = Thai2fitAug()
        text = "ประกาศมหาวิทยาลัย ฉบับที่ 6"
        self.assertIsNotNone(w2v.augment(text))
