# -*- coding: utf-8 -*-
from datasets import load_dataset

#transformers
from transformers import (
    CamembertTokenizer,
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import random
#thai2transformers
from typing import List
import thai2transformers
from thai2transformers.preprocess import process_transformers

model_name = "airesearch/wangchanberta-base-att-spm-uncased"


target_tokenizer = CamembertTokenizer
tokenizer = CamembertTokenizer.from_pretrained(
                                    model_name ,
                                    revision='main')
tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', '<_>']

#pipeline
fill_mask = pipeline(task='fill-mask',
         tokenizer=tokenizer,
         model = f'{model_name}',
         revision = 'main',)


class Thai2transformersAug:
    def __init__(self):
        self.model_name = "airesearch/wangchanberta-base-att-spm-uncased"
        self.target_tokenizer = CamembertTokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained(
                                    self.model_name,
                                    revision='main')
        self.tokenizer.additional_special_tokens = ['<s>NOTUSED', '</s>NOTUSED', '<_>']
        self.fill_mask = pipeline(
            task='fill-mask',
            tokenizer=self.tokenizer,
            model = f'{self.model_name}',
            revision = 'main',)
    def generate(self, sentence: str, num_replace_tokens: int=3):
        self.input_text = process_transformers(sentence)
        sent = tokenizer.tokenize(self.input_text)
        if len(list(set(sent))) < num_replace_tokens:
            num_replace_tokens = len(list(set(sent)))
        masked_text = self.input_text
        for i in range(num_replace_tokens):
            replace_token = random.choice(sent)
            masked_text = masked_text.replace(replace_token, f"{self.fill_mask.tokenizer.mask_token}",1)
            self.sent2+=[j['sequence'] for j in self.fill_mask(masked_text+'<pad>')]
            masked_text = self.input_text
            sent = tokenizer.tokenize(self.input_text)
        return self.sent2

    def augment(self, sentence: str, num_replace_tokens: int=3) -> List[str]:
        """
        Text Augment from wangchanberta

        :param str sentence: thai sentence
        :param int num_replace_tokens: number replace tokens

        :return: list of text augment
        :rtype: List[str]
        """
        self.sent2 = []
        try:
            return self.generate(sentence, num_replace_tokens)
        except:
            if len(self.sent2) > 0:
                return self.sent2
            else:
                return self.sent2