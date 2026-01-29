#torchrun --nproc_per_node=6 train.py 
import os
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import time
import datetime
import torch
import torch.nn.functional as F
from scripts.Logger import Logger1
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from scripts.lossfun import Fusionloss,Loss_seg,L_SSIM,Percep_Loss,PixelLoss
import numpy as np
from dataload.FMB import FMBDataSet
import argparse
import warnings
from common import utils
import logging
import shutil
import re
from Net import net
from scripts.utils import read_data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings('ignore')
local_rank = int(os.environ.get('LOCAL_RANK', -1))
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.getcwd())
logging.basicConfig(level=logging.CRITICAL)
parser = argparse.ArgumentParser()
parser.add_argument('--numepochs', type=int, default=200, help='')
parser.add_argument('--lr', type=float, default=1e-4, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--loss_ingrad_weight', type=int, default=1, help='')
parser.add_argument('--loss_pixel_weight', type=int, default=1, help='')
parser.add_argument('--loss_seg_weight', type=int, default=1, help='')
parser.add_argument('--loss_percep_weight', type=int, default=1, help='')
parser.add_argument('--dataset_path', type=str, default="/root/dataset/FMB", help='')###########################################Your dataset path
parser.add_argument('--backbone', type=str, default='sam')
parser.add_argument('--condition', type=str, default='mask')
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--num_query', type=int, default=50)
parser.add_argument('--pre_ckt', type=str, default='')
opt = parser.parse_args()
# Distributed setting
dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)    

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="",  ################################################Your Project Name
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "dataset": "FMB",
    "epochs": 200,
    }
)
'''
------------------------------------------------------------------------------
Set the hyper-parameters for training
------------------------------------------------------------------------------
'''
pre_ckt = opt.pre_ckt
pre_model = ""
epochs = opt.numepochs
lr = opt.lr
weight_decay = opt.weight_decay
batch_size = opt.batch_size
weight_ingrad = opt.loss_ingrad_weight
weight_pixel = opt.loss_pixel_weight
weight_seg=opt.loss_seg_weight
weight_percep = opt.loss_percep_weight
dataset_path = opt.dataset_path
exp_name = ''
optim_step = 20
optim_gamma = 0.5
'''
------------------------------------------------------------------------------
model
------------------------------------------------------------------------------
'''
############################################Data preparation########################################
train_path_list, val_path_list = read_data(dataset_path)
train_dataset=FMBDataSet(train_path_list,phase="train")
val_dataset=FMBDataSet(val_path_list,phase="val")
trainloader = DataLoader(train_dataset,
                            batch_size=opt.batch_size,
                            pin_memory=True,
                            num_workers=0,
                            collate_fn=train_dataset.collate_fn,
                            sampler=DistributedSampler(train_dataset))
valloader = DataLoader(val_dataset,
                            batch_size=1,
                            pin_memory=True,
                            num_workers=0,
                            collate_fn=val_dataset.collate_fn,
                            sampler=DistributedSampler(val_dataset))
##########################################model##################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = net(opt)######SAMM2pred
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
start_epoch=0
###########################################optimizer######################################
for param in model.module.sam_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
############################################scheduler#####################################
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optim_step, gamma=optim_gamma)
###############################################################save#########################################
filefold_path = "xxxx"#######################Your save path
if not os.path.exists(filefold_path): os.makedirs(filefold_path,exist_ok=True)
file_img_path = os.path.join(filefold_path, "img")
if not os.path.exists(file_img_path): os.makedirs(file_img_path,exist_ok=True)
file_weights_path = os.path.join(filefold_path,'weight')
if not os.path.exists(file_weights_path): os.makedirs(file_weights_path,exist_ok=True)
#file_grad_path = os.path.join(filefold_path,'grad')
#if not os.path.exists(file_grad_path): os.makedirs(file_grad_path,exist_ok=True)
file_log_path = os.path.join(filefold_path, "log")
if not os.path.exists(file_log_path): os.makedirs(file_log_path,exist_ok=True)
logger = Logger1(rootpath=file_log_path, timestamp=False)
params = {
    'epoch': epochs,
    'lr': lr,
    'weight_ingrad':weight_ingrad,
    'weight_seg':weight_seg,
    'weight_percep':weight_percep,
    'weight_decay':weight_decay 
}
logger.save_param(params)
writer = SummaryWriter(logger.logpath)
#############################################Load a pre-trained model##############################################3
if pre_ckt !="":
    model_weight_path = pre_ckt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = net().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    optimizer.load_state_dict(torch.load(model_weight_path, map_location=device)['optimizer'])
    scheduler.load_state_dict(torch.load(model_weight_path, map_location=device)['lr_schedule'])
    start_epoch = torch.load(model_weight_path, map_location=device)['epoch']  + 1
    print('load_pretrain_model')
'''
------------------------------------------------------------------------------
Train Val
------------------------------------------------------------------------------
'''
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
start_time = time.time()
Loss_ingrad = Fusionloss().to(device)
Loss_ssim = L_SSIM().to(device)
Loss_seg = Loss_seg().to(device)
Loss_percep = Percep_Loss().to(device)
Loss_pixel = PixelLoss().to(device)
best_val_loss=1e10
####################################################T&V###############################################################
for epoch in range(start_epoch,epochs):
    s_temp = time.time()
    savefold_path = os.path.join(file_img_path, str(epoch))
    if os.path.exists(savefold_path) is False:
            os.makedirs(savefold_path,exist_ok=True)
######################################################Train################################################
    for i, data in enumerate(trainloader):
        model.module.train()
        I_A, I_B, mask, name = data
        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            mask = mask.to(device)
        
        I_ref,logit_mask_v,logit_mask_i,pred_mask_v,pred_mask_i,pred_mask,I_fused =model(opt.condition, I_A, I_B, mask, name)
        batchsize, channels, rows, columns = I_A.shape
        I_A = I_A.mean(dim=1, keepdim=True)
        I_B = I_B.mean(dim=1, keepdim=True)
    #######################################loss calculation###########################################################
        loss_seg = Loss_seg(logit_mask_v, mask)*0.5+Loss_seg(logit_mask_i, mask)*0.5
        loss_ingrad1= Loss_ingrad(I_A, I_B, I_ref)
        loss_ingrad2= Loss_ingrad(I_A, I_B, I_fused)
        loss_ingrad = loss_ingrad1 + loss_ingrad2
        loss_percep = Loss_percep(I_A,I_fused)*0.5+Loss_percep(I_B,I_fused)*0.5
        loss_pixel = Loss_pixel(I_fused,I_A,I_B,mask)
        lossALL = loss_ingrad*weight_ingrad + loss_percep*weight_percep + loss_seg*weight_seg +loss_pixel*weight_pixel
        # log metrics to wandb
        wandb.log({"lossALL": lossALL,'loss_ingrad1':loss_ingrad1,'loss_ingrad2':loss_ingrad2,'loss_percep':loss_percep,"loss_seg":loss_seg,'loss_pixel':loss_pixel})
    ################################################################################################
        optimizer.zero_grad()  
        lossALL.backward()
        #with open(file_grad_path+"/grad.txt", "a") as f:
        #    f.write(f"##########################Epoch {epoch}, Batchsize {i}##################################\n")
        #    for name, param in model.named_parameters():
        #        if param.requires_grad == True:
        #            f.write(f"Parameter: {name}, Gradient: {param.grad}\n")
        optimizer.step()
        batches_done = epoch * len(trainloader) + i
        batches_left = epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        logger.log_and_print(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss_ingrad: %f] [loss_percep: %f] [loss_seg: %f] [loss_pixel: %f]"
            % (
                epoch + 1,
                epochs,
                i,
                len(trainloader),
                lossALL.item(),
                loss_ingrad.item(),
                loss_percep.item(),
                loss_seg.item(),
                loss_pixel.item()
            )
        )
        writer.add_scalar('loss/01 Loss', lossALL.item(), step)
        writer.add_scalar('loss/01 loss_ingrad', loss_ingrad.item(), step)
        writer.add_scalar('loss/01 loss_percep', loss_percep.item(), step)
        writer.add_scalar('loss/01 loss_seg', loss_seg.item(), step)
        writer.add_scalar('loss/01 loss_pixel', loss_pixel.item(), step)
        writer.add_scalar('loss/14 learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step += 1
        torch.cuda.empty_cache() 
    scheduler.step()
#####################################################val#####################################################
    with torch.no_grad():
        model.module.eval()
        accu_total_valloss=0
        for i, data in enumerate(valloader):
                I_A, I_B, mask, name = data
                if torch.cuda.is_available():
                   I_A = I_A.to(device)
                   I_B = I_B.to(device)
                   mask = mask.to(device)
                I_ref,logit_mask_v,logit_mask_i,pred_mask_v,pred_mask_i,pred_mask,I_fused =model(opt.condition, I_A, I_B, mask, name)
                batchsize, channels, rows, columns = I_A.shape
                I_A = I_A.mean(dim=1, keepdim=True)
                I_B = I_B.mean(dim=1, keepdim=True)
                val_loss_seg = Loss_seg(logit_mask_v, mask)*0.5 + Loss_seg(logit_mask_i, mask)*0.5
                val_loss_ingrad1 = Loss_ingrad(I_A, I_B, I_ref)
                val_loss_ingrad2 = Loss_ingrad(I_A, I_B, I_fused)
                val_loss_ingrad = val_loss_ingrad1 + val_loss_ingrad2
                val_loss_percep = Loss_percep(I_A,I_fused)*0.5 + Loss_percep(I_B,I_fused)*0.5
                val_loss_pixel = Loss_pixel(I_fused,I_A,I_B,mask)
                val_lossALL = val_loss_ingrad*weight_ingrad + val_loss_percep*weight_percep + val_loss_seg*weight_seg +val_loss_pixel*weight_pixel
                logger.log_and_print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss_ingrad: %f] [loss_percep: %f] [loss_seg: %f] [loss_pixel: %f]"
                % (
                epoch + 1,
                epochs,
                i,
                len(valloader),
                val_lossALL.item(),
                val_loss_ingrad.item(),
                val_loss_percep.item(),
                val_loss_seg.item(),
                val_loss_pixel.item()
                ))
                wandb.log({"val_lossALL": val_lossALL,'val_loss_ingrad1':val_loss_ingrad1,'val_loss_ingrad2':val_loss_ingrad2,'val_loss_percep':val_loss_percep,'val_loss_seg':val_loss_seg,'val_loss_pixel':val_loss_pixel})
                accu_total_valloss=accu_total_valloss+val_lossALL
                batchsize, channels, rows, columns = I_A.shape
                if i<40:
                    for j in range(batchsize):
                        fi = np.squeeze(I_fused[j].detach().cpu().numpy()) * 255
                        i_ref=np.squeeze(I_ref[j].detach().cpu().numpy()) * 255
                        seg_pred_mask=np.squeeze(pred_mask[j].detach().cpu().numpy()) * 255
                        seg_pred_mask_v=np.squeeze(pred_mask_v[j].detach().cpu().numpy()) * 255
                        seg_pred_mask_i=np.squeeze(pred_mask_i[j].detach().cpu().numpy()) * 255
                        plt.imsave(os.path.join(savefold_path, str(name[j]) + 'val.png'), fi, cmap="gray")
                        plt.imsave(os.path.join(savefold_path, str(name[j]) + 'ref.png'), i_ref, cmap="gray")
                        plt.imsave(os.path.join(savefold_path, str(name[j]) + 'predmask.png'), seg_pred_mask, cmap="gray")
                        plt.imsave(os.path.join(savefold_path, str(name[j]) + 'predmask_v.png'), seg_pred_mask_v, cmap="gray")
                        plt.imsave(os.path.join(savefold_path, str(name[j]) + 'predmask_i.png'), seg_pred_mask_i, cmap="gray")
    logger.log_and_print("accu_total_valloss: %f " % (accu_total_valloss.item()))
    wandb.log({"accu_total_valloss:": accu_total_valloss})
    if accu_total_valloss.item()<best_val_loss:
        checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                "epoch": epoch,
            }
        torch.save(checkpoint, file_weights_path + "/" + "checkpoint_best.pth")
        best_val_loss = accu_total_valloss.item()
    e_temp = time.time()
    print("This Epoch takes time: " + str(e_temp - s_temp))
end_time = time.time()
logger.log_and_print("total_time: " + str(end_time - start_time))
