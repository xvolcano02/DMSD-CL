import os
import argparse
import time
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch import optim

from data_utils import RGDataset,RG2Dataset
from model import DMSD
import torch.nn.functional as F
from losses import SupConLoss2
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy,macro_f1,weighted_f1
from util import set_optimizer, save_model
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] ="5"
log_dir = 'logs/tensorboard/main'
save_folder = 'saved_models/main'
batch_size=16
writer = SummaryWriter(log_dir)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument("--data_path",default= '../data',type=str)
    parser.add_argument("--image_path",default='../data/dataset_image',type=str)
    parser.add_argument("--save_folder",default=save_folder,type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_len",default=100,type=int,
                        help="Total number of text.can not alter")
    
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch_size')
    parser.add_argument('--seed',type=int,default=777,
                        help="random seed for initialization")
    parser.add_argument("--alpha", default= '0.1',
                        type=float)
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=70,
                        help='number of training epochs')
    # optimization
    parser.add_argument("--global_learning_rate",
                        default=1e-4,
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
    cl_criterion = SupConLoss2(temperature=opt.temp,t=0.8)

    if torch.cuda.is_available():
        model = model.cuda()
        ce_criterion = ce_criterion.cuda()
        cl_criterion = cl_criterion.cuda()
        cudnn.benchmark = True

    return model, ce_criterion, cl_criterion

def train(train_loader, model, ce_criterion, cl_criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, batch in enumerate(tqdm(train_loader)):
        bert_mask,bert_mask_add, bert_indices_add, images,labels=batch
        
        bert_mask=torch.cat([bert_mask[0], bert_mask[1],bert_mask[2]], dim=0)
        bert_mask_add=torch.cat([bert_mask_add[0], bert_mask_add[1],bert_mask_add[2]], dim=0)
        bert_indices_add=torch.cat([bert_indices_add[0], bert_indices_add[1],bert_indices_add[2]], dim=0)
        images = torch.cat([images[0], images[1],images[2]], dim=0)  #[bsz*3,3,224,224]
        labels = torch.cat([labels[0], labels[1],labels[2]], dim=0)  #[bsz*3,3,224,224]
        if torch.cuda.is_available():
            bert_mask=bert_mask.cuda(non_blocking=True)
            bert_mask_add=bert_mask_add.cuda(non_blocking=True)
            bert_indices_add=bert_indices_add.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)         #[bsz*3,1]
        bsz = labels.shape[0]

        # compute loss
        output, features = model(bert_mask,bert_mask_add, bert_indices_add, images,"train")    #[bsz*3,feature_dim]
        ce_loss = ce_criterion(output,labels)
        cl_loss = cl_criterion(features,labels)
        loss = opt.alpha*ce_loss+(1-opt.alpha)*cl_loss 
        # update metric
        losses.update(loss.item(), bsz)
        acc1= accuracy(output, labels)
        top1.update(acc1[0], bsz)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, top1.avg

def eval(val_loader, model, ce_criterion, opt):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    y_true=[]
    y_pred=[]
    with torch.no_grad():
        
        for idx, batch in enumerate(tqdm(val_loader)):
            bert_mask,bert_mask_add, bert_indices_add, images,labels=batch
            y_true.append(labels.numpy())
            if torch.cuda.is_available():
                bert_mask=bert_mask.cuda()
                bert_mask_add=bert_mask_add.cuda()
                bert_indices_add=bert_indices_add.cuda()
                images = images.cuda()
                labels = labels.cuda()         #[bsz,1]
            bsz = labels.shape[0]

            # forward
            output = model(bert_mask,bert_mask_add, bert_indices_add, images,'test')
            loss = ce_criterion(output, labels)
            y_pred.append(output.to('cpu').numpy())
            # update metric
            losses.update(loss.item(), bsz)
            acc1= accuracy(output, labels)
            top1.update(acc1[0], bsz)
        y_true=np.concatenate(y_true)
        y_pred=np.concatenate(y_pred)
        precision, recall, F_score = macro_f1(y_true, y_pred)
        w_pre,w_rec,w_f1 = weighted_f1(y_true, y_pred)
            
    return losses.avg, top1.avg, precision, recall, F_score, w_pre,w_rec,w_f1

def main():
    best_valid_acc = 0
    best_test_acc = 0
    best_ood_acc=0
    opt = parse_option()
    print(opt)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    random.seed(opt.seed)
    # build data loader
    train_loader, valid_loader, test_loader, ood_loader = set_loader(opt)

    # build model and criterion
    model, ce_criterion, cl_criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, ce_criterion, cl_criterion,optimizer, epoch, opt)
        time2 = time.time()
        writer.add_scalar('train_loss', loss, global_step=epoch)
        writer.add_scalar('train_acc', acc, global_step=epoch)
        print(f'Train epoch {epoch}, total time {time2 - time1}, train_loss:{loss}, train_accuracy:{acc}')
        
        loss, val_acc, precision, recall, F_score, w_pre,w_rec,w_f1 = eval(valid_loader, model, ce_criterion, opt)
        print(f'Train epoch {epoch}, valid_loss:{loss}, valid_accuracy:{val_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_valid0"  + ".bin"))
            print("better model")
       # eval for one epoch
        loss, test_acc, precision, recall, F_score, w_pre,w_rec,w_f1 = eval(test_loader, model, ce_criterion, opt)
        writer.add_scalar('test_loss', loss, global_step=epoch)
        writer.add_scalar('test_acc', test_acc, global_step=epoch)
        writer.add_scalar('F_score', F_score, global_step=epoch)
        writer.add_scalar('w_f1', w_f1, global_step=epoch)
        print(f'Train epoch {epoch}, test_loss:{loss}, test_accuracy:{test_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_test0"  + ".bin"))
            print("better model")
        loss, ood_acc, precision, recall, F_score, w_pre,w_rec,w_f1 = eval(ood_loader, model, ce_criterion, opt)
        writer.add_scalar('OOD_loss', loss, global_step=epoch)
        writer.add_scalar('OOD_acc', ood_acc, global_step=epoch)
        print(f'Train epoch {epoch}, OOD_loss:{loss}, OOD_accuracy:{ood_acc} \nprecision:{precision}, recall:{recall},F_score:{F_score}\nw_pre:{w_pre}, w_rec:{w_rec}, w_f1:{w_f1}')
        if ood_acc > best_ood_acc:
            best_ood_acc = ood_acc
            torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_OOD0"  + ".bin"))
            print("better model")
            
    torch.save(model.state_dict(), os.path.join(opt.save_folder, "pytorch_model_aver_last"  + ".bin"))
    print('best accuracy: {:.2f}'.format(best_test_acc))
    print('best accuracy: {:.2f}'.format(best_ood_acc))
      

if __name__ == "__main__":
    main()



