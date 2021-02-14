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
#thai2transformers
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
    def fill_mask(self, input_text):
        self.fill_mask = pipeline(
            task='fill-mask',
            tokenizer=self.tokenizer,
            model = f'{self.model_name}',
            revision = 'main',)
        self.input_text = process_transformers(input_text)
        return self.fill_mask(input_text+'<pad>')