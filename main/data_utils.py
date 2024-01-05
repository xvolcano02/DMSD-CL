import os
import pickle
from matplotlib.pyplot import text
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import RobertaTokenizer
from util import text_aug,get_opposite,get_train,get_valid,get_test,get_ood
from torch.utils.data import DataLoader
import random
import pandas as pd

tokenizer = RobertaTokenizer.from_pretrained('../model/roberta-base')


class RGDataset(Dataset):
    def __init__(self, id_path,image_path,max_len):
        with open(id_path, "r") as f:
            self.id = eval(f.read())
            self.image_path = "/data/jingliqiang/xiecan/data/sarcasm-detection/dataset_image"
            self.all_text_dic, self.all_label_dic = get_train()
            self.opposite_text_dic = get_opposite()
            self.max_len = max_len

    # 图片转tensor
    def image_process(self,image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        
        
        image = Image.open(image_path)
        image = transform(image)
        return image

    # bert_tokeninzer 处理成input_id，mask
    def text_process(self,text_now):
        tokens = text_now
        bert_indices = tokenizer(tokens)['input_ids']
        if len(bert_indices) > self.max_len:
            bert_indices = bert_indices[:(self.max_len - 1)]
            bert_indices.append(2)
        len_text = len(bert_indices)
        len_pad = self.max_len - len_text
        mask = []
        for m in range(len_text):
            mask.append(1)
        for m in range(len_pad):
            mask.append(0)
            bert_indices.append(1)
        bert_mask = torch.tensor(mask)
        bert_indices = torch.tensor(bert_indices)
        return bert_mask, bert_indices


    def __getitem__(self, index):
        name = self.id[index]
        label1 = self.all_label_dic[name]
        label2 = label1
        label3 = label1^1
        label=[label1,label2,label3]

        text_raw = self.all_text_dic[name]
        
        pos_id = name
        text_pos = text_aug(text_raw)

        neg_id = name
        text_neg = self.opposite_text_dic[neg_id]

        names = [name,pos_id,neg_id]
        text = [text_raw,text_pos,text_neg]
        
        image_paths = [os.path.join(self.image_path, str(names[0]) + ".jpg"),
                       os.path.join(self.image_path, str(names[1]) + ".jpg"),
                       os.path.join(self.image_path, str(names[2]) + ".jpg")]
            
        image_trans=[]
        bert_mask=[]
        bert_indices=[]

        for i in range(len(names)):
            image_text = ''
            mask, indices = self.text_process(text[i])
            bert_mask.append(mask)
            bert_indices.append(indices)
            image_trans.append(self.image_process(image_paths[i]))
        
        # return bert_mask, bert_mask_add, bert_indices_add,image_trans,label
        return bert_mask, bert_mask, bert_indices,image_trans,label

    def __len__(self):
        return len(self.id)
    
class RG2Dataset(Dataset):
    def __init__(self, id_path,image_path,max_len):
        with open(id_path, "r") as f:
            self.id = eval(f.read())
            self.image_path = image_path
            if 'valid' in id_path:
                self.all_text_dic, self.all_label_dic= get_valid()
            else:
                self.all_text_dic, self.all_label_dic= get_test()
            # self.image_text_dic = get_image_text()
            self.ood_text_dic,self.ood_label_dic = get_ood()

            self.max_len = max_len

    # 图片转tensor
    def image_process(self,image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        image = Image.open(image_path)
        image = transform(image)
        return image

    # bert_tokeninzer 处理成input_id，mask
    def text_process(self,text_now):
        tokens = text_now
        bert_indices = tokenizer(tokens)['input_ids']
        if len(bert_indices) > self.max_len:
            bert_indices = bert_indices[:(self.max_len - 1)]
            bert_indices.append(2)
        len_text = len(bert_indices)
        len_pad = self.max_len - len_text
        mask = []
        for m in range(len_text):
            mask.append(1)
        for m in range(len_pad):
            mask.append(0)
            bert_indices.append(1)
        bert_mask = torch.tensor(mask)
        bert_indices = torch.tensor(bert_indices)
        return bert_mask, bert_indices


    def __getitem__(self, index):
        name = self.id[index]
        if 'o' in name:
            label = self.ood_label_dic[name]
            text_raw = self.ood_text_dic[name]
            name=name.split('o')[1]
        else:
            label = self.all_label_dic[name]
            text_raw = self.all_text_dic[name]
            
        bert_mask, bert_indices = self.text_process(text_raw)

        # image trans
        image_path = os.path.join(self.image_path, str(name) + ".jpg")
        image_trans = self.image_process(image_path)  # 3*224*224p

        # return bert_mask, bert_mask_add, bert_indices_add,image_trans,label
        return bert_mask, bert_mask, bert_indices,image_trans,label

    def __len__(self):
        return len(self.id)  
    
      
if __name__ == '__main__':
    a,b = get_test()
