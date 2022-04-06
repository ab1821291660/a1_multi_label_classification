# -*- coding: utf-8 -*-
import torch
from a0data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import BertTokenizer
hidden_size = 768
class_num = 3
label2idx_path = "./data/label2idx.json"
save_model_path = "./model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
idx2label = {idx: label for label, idx in label2idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("./model/roberta_zh")##8888##===================================
max_len = 128

model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
model.eval()


# tokenizers = self.tokenizer(texts,
#                             padding=True,
#                             truncation=True,
#                             max_length=self.max_len,
#                             return_tensors="pt",
#                             is_split_into_words=False)
# input_ids = tokenizers["input_ids"]
# token_type_ids = tokenizers["token_type_ids"]
# attention_mask = tokenizers["attention_mask"]
def predict(texts):
    outputs = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")##===================================
    logits = model(outputs["input_ids"],
                   outputs["token_type_ids"],
                   outputs["attention_mask"])
    logits = logits.cpu().tolist()##===================================
    print(logits)#b2-c
    #
    result = []
    for sample in logits:
        pred_label = []
        for idx, logit in enumerate(sample):
            if logit > 0.5:
                pred_label.append(idx2label[idx])
        result.append(pred_label)
    return result
if __name__ == '__main__':
    texts = ["xxxx",
             "yyyy"]#[['体育'], ['体育', '中超']]
    result = predict(texts)
    print(result)



