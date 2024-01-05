import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss


class SupConLoss2(nn.Module):
    def __init__(self, temperature=0.07,eps = 0.5,t=1.00):
        super(SupConLoss2, self).__init__()
        self.temperature = temperature
        self.eps = torch.tensor(eps)
        self.t = t

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_feature = features

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, anchor_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        similarity = torch.matmul(anchor_feature, anchor_feature.T)
        #自适应权重分配
        mask_same_label =  mask - torch.eye(batch_size).to(device)     #without self
        mask_diff_label = 1 - mask
        #相同的标签越不相似权重越高
        for i in range(batch_size):
            sample = similarity[i]
            index1 = mask_same_label[i]!=0
            sample[index1] = 1.0 - torch.softmax(sample[index1],0)
            index2 = mask_diff_label[i]!=0
            sample[index2] = 1.0 + torch.softmax(sample[index2],0)
        w_same = similarity*mask_same_label / self.t
        w_diff = similarity*mask_diff_label / self.t
        logits_mask = w_same + w_diff
    
        
        mask = mask_same_label
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = - (log_prob*w_same).sum(1) / mask.sum(1)

        # loss
        loss = mean_log_prob_pos.mean()
        return loss
    
if __name__ == '__main__':
    import random
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    loss = SupConLoss2(t=0.8)
    # loss = SupConLoss(eps=0.0)
    features = torch.rand(12,8)
    labels = torch.tensor([1,0,0,1,0,1,1,0,0,1,1,0])
    l = loss(features,labels)
    print(l)
    
