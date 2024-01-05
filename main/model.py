import math
import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel
from collections import Counter
from model.PyTorch_Pretrained_ViT.pytorch_pretrained_vit.model import ViT
import pickle
import tqdm
import os
from torch.utils.data import DataLoader

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertPooler_i2t(nn.Module):
    def __init__(self):
        super(BertPooler_i2t, self).__init__()
        self.dense = nn.Linear(768,768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPooler(nn.Module):
    def __init__(self):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]   
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCoAttention(nn.Module):
    def __init__(self):
        super(BertCoAttention, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        # s2_attention_mask  b*1*1*49
        mixed_query_layer = self.query(s1_hidden_states)  # b*75*768
        mixed_key_layer = self.key(s2_hidden_states)  # b*49*768
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # b*12*75*64
        key_layer = self.transpose_for_scores(mixed_key_layer)  # b*12*49*64
        value_layer = self.transpose_for_scores(mixed_value_layer)  # b*12*49*64

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1, -2))  # b*12*75*49
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # b*12*75*49
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask
        # attention_scores b*12*75*49
        # Normalize the attention scores to probabilities.
        # b*12*75*49
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # aa = attention_probs.cpu().numpy()
        # aa = numpy.sum(aa,axis=1,keepdims=True)
        # aa = numpy.sum(aa,axis=2,keepdims=True)
        # aa = numpy.linalg.norm(aa, ord=None, axis=-2, keepdims=True)
        # with open('/data/qiaoyang/ms_data/aa.pkl', 'wb') as f:
        #     pickle.dump(aa, f)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)      # (32,12,197,64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()      # (32,197,12,64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)     # (32,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer b*75*768
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossAttention(nn.Module):
    def __init__(self):
        super(BertCrossAttention, self).__init__()
        self.bertCoAttn = BertCoAttention()
        self.output = BertSelfOutput()

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.bertCoAttn(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self):
        super(BertCrossAttentionLayer, self).__init__()
        self.bertCorssAttn = BertCrossAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.bertCorssAttn(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        # b*75*768
        intermediate_output = self.intermediate(attention_output)
        # b*75*3072
        layer_output = self.output(intermediate_output, attention_output)
        # b*75*3072
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(3)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        return s1_hidden_states



class DMSD(nn.Module):
    def __init__(self,feature_dim=128):
        super(DMSD, self).__init__()
        self.roberta = RobertaModel.from_pretrained('../model/roberta-base', return_dict=True)
        self.text_image_attention = BertCrossEncoder()
        self.dropout = nn.Dropout(0.1)
        self.cls_pooler = BertPooler()
        self.classifier = nn.Linear(768, 2)
        self.vit = ViT('B_16',pretrained=True)
        self.projection=nn.Sequential(nn.Linear(768, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, feature_dim))

    def forward(self,input_mask,bert_mask_add, bert_indices_add, img_trans,mode="train"):
        outputs = self.roberta(input_ids=bert_indices_add, attention_mask=bert_mask_add)
        pooler_output = outputs.last_hidden_state   #pooler_output(batchsize,77,768)
        mean, visual = self.vit(img_trans)      # viasual(batchsoze,197,768)

        text_mask = input_mask.to(dtype=torch.float32)  #text_mask(batchsize,77)
        text_mask = (1.0 - text_mask) * -10000.0
        text_extended_mask = text_mask.unsqueeze(1).unsqueeze(2)   # (batchsize,1,1,77)

        cross_attn = self.text_image_attention(visual,pooler_output,text_extended_mask)  #(batchsize,197,768)

        cls = self.cls_pooler(cross_attn)  #(batchsize,768)
        pooled_output=(self.dropout(cls))
        if mode=="train":
            output = self.classifier(pooled_output)
            feature=F.normalize(self.projection(cls), dim=1)
            return output,feature
        else:
            output = self.classifier(pooled_output)
            return output

if __name__ == '__main__':
    data=torch.rand(2,128)
    data2=torch.rand(2,128)
    print(torch.cat([data,data2],dim=1).shape)
    