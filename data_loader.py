# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from a0data_preprocess import load_json


class MultiClsDataSet(Dataset):
    def __init__(self, data_path, max_len=128, label2idx_path="./data/label2idx.json"):
        self.label2idx = load_json(label2idx_path)
        self.class_num = len(self.label2idx)
        # self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")##8888##===================================
        self.tokenizer = BertTokenizer.from_pretrained("./model/roberta_zh")
        self.max_len = max_len
        self.input_ids, self.token_type_ids, self.attention_mask, self.labels = self.encoder(data_path)
    def encoder(self, data_path):
        texts = []
        labels = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                texts.append(line["text"])
                tmp_label = [0] * self.class_num
                for label in line["label"]:
                    tmp_label[self.label2idx[label]] = 1#原1报错RuntimeError: Found dtype Int but expected Float##8888##===================================
                labels.append(tmp_label)
        tokenizers = self.tokenizer(texts,
                                    padding=True,
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt",
                                    is_split_into_words=False)
        input_ids = tokenizers["input_ids"]
        token_type_ids = tokenizers["token_type_ids"]
        attention_mask = tokenizers["attention_mask"]
        return input_ids, token_type_ids, attention_mask, torch.tensor(labels, dtype=torch.float)#原dtype=torch.int32 #不用改为float ##8888##===================================
        #torch.float32或torch.float：32位浮点
        #torch.float64或torch.double：64位，双精度浮点
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        return self.input_ids[item], self.token_type_ids[item], self.attention_mask[item], self.labels[item]



