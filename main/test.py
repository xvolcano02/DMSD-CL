import os
import argparse
import time
from unittest import result
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import random
import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch import optim

from data_utils import RGDataset,RG2Dataset
from model_contrastive import DMSD

import torch.nn.functional as F
from losses import SupConLoss
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy,macro_f1,weighted_f1
from util import set_optimizer, save_model
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
#每次改的参数
os.environ["CUDA_VISIBLE_DEVICES"] ="3"
# log_dir = '/data/jingliqiang/xiecan/ConSam/logs/tensotboard/simple_ce3'
save_folder = '/data/jingliqiang/xiecan/ConSam2/SupCon/saved_models/main'
batch_size=16
# writer = SummaryWriter(log_dir)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument("--data_path",default= '/data/jingliqiang/xiecan/ConSam2/data',type=str)
    parser.add_argument("--image_path",default='/data/jingliqiang/xiecan/data/sarcasm-detection/dataset_image',type=str)
    parser.add_argument("--save_folder",default=save_folder,type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_len",default=77,type=int,
                        help="Total number of text.can not alter")
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='batch_size')
    parser.add_argument('--seed',type=int,default=777,
                        help="random seed for initialization")
    parser.add_argument("--alpha", default= '0.1',
                        type=float)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=70,
                        help='number of training epochs')
    # optimization
    parser.add_argument("--global_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--global_pre",
                        default=1e-5, 
                        type=float)
    parser.add_argument("--global_weight_decay",
                        default=1e-5,
                        type=float)
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30,40,50',
                        help='where to decay lr, can be a list')
    
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # other setting
    parser.add_argument('--cosine', action='store_true',default=True,
                        help='using cosine annealing')

    opt = parser.parse_args()
    
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    return opt

def set_loader(opt):
    # construct data loader
    train_dataset = RGDataset(os.path.join(opt.data_path, 'train_id.txt'),opt.image_path, opt.max_len)
    valid_dataset = RG2Dataset(os.path.join(opt.data_path, 'valid_id.txt'),opt.image_path, opt.max_len)
    test_dataset = RG2Dataset(os.path.join(opt.data_path, 'test_id.txt'),opt.image_path, opt.max_len)
    ood_dataset = RG2Dataset(os.path.join(opt.data_path, 'ood_id.txt'),opt.image_path, opt.max_len)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    
    ood_loader = DataLoader(ood_dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    return train_loader,valid_loader,test_loader,ood_loader

def set_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMSD()
    ce_criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        ce_criterion = ce_criterion.cuda()
        cudnn.benchmark = True

    return model, ce_criterion


def test():
    opt = parse_option()
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)

    train_loader, valid_loader, test_loader, ood_loader = set_loader(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMSD()

    model.load_state_dict(torch.load(os.path.join(opt.save_folder, "pytorch_model_aver_valid0"  + ".bin")))
    model.to(device)
    """test"""
    model.eval()

    top1 = AverageMeter()
    ids=[]
    y_pred_label=[]
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        
        for idx, batch in enumerate(tqdm(test_loader)):
            name,bert_mask,bert_mask_add, bert_indices_add, images,labels=batch
            # bert_mask,bert_mask_add, bert_indices_add, images,labels=batch
            ids.append(name)
            y_true.append(labels.numpy())
            if torch.cuda.is_available():
                bert_mask=bert_mask.cuda()
                bert_mask_add=bert_mask_add.cuda()
                bert_indices_add=bert_indices_add.cuda()
                images = images.cuda()
                labels = labels.cuda()         #[bsz,1]
            bsz = labels.shape[0]

            # forward
            
            output = model(bert_mask,bert_mask_add, bert_indices_add, images,"test")
            y_pred.append(output.to('cpu').numpy())
            # update metric
            acc1= accuracy(output, labels)
            top1.update(acc1[0], bsz)
        ids=np.concatenate(ids)
        y_true=np.concatenate(y_true)
        y_pred=np.concatenate(y_pred)
        y_pred_label = np.argmax(y_pred, axis=-1)
        precision, recall, F_score = macro_f1(y_true, y_pred)
        w_pre,w_rec,w_f1 = weighted_f1(y_true, y_pred)
        result={
            'ids':ids,
            "y_output":y_pred,
            'y_pred_label':y_pred_label,
            'ground-truth':y_true,
            'f1':[precision,recall,F_score],
            'w_f1':[w_pre,w_rec,w_f1]
        }
        with open("/data/jingliqiang/xiecan/ConSam2/SupCon/test-detail.pkl","wb") as f:
            pickle.dump(result,f)
            
    return top1.avg, precision, recall, F_score, w_pre,w_rec,w_f1
      

if __name__ == "__main__":
    test_acc,precision, recall, F_score, w_pre,w_rec,w_f1=test()
    print(f'test_accuracy:{test_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')



