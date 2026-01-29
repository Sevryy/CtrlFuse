import os
import sys
import random
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scripts.lossfun import Fusionloss, LpLssimLossweight
import torchvision.transforms as transforms
from PIL import Image
from common import utils
def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)########/root/project/FMB

    train_root = os.path.join(root, "trn")#######
    val_root = os.path.join(root, "val")###########
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_mask_path=[]
    val_images_visible_path = []
    val_images_infrared_path = []
    val_images_mask_path=[]

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  

    train_visible_root = os.path.join(train_root, "Visible")#######
    train_infrared_root= os.path.join(train_root, "Infrared")#######
    train_mask_root= os.path.join(train_root, "mask")#######

    val_visible_root = os.path.join(val_root, "Visible")##########
    val_infrared_root = os.path.join(val_root, "Infrared")#########
    val_mask_root= os.path.join(val_root, "mask")#######
    
    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]
    train_mask_path = [os.path.join(train_mask_root, i) for i in os.listdir(train_mask_root)
                  if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                  if os.path.splitext(i)[-1] in supported]
    val_mask_path = [os.path.join(val_mask_root, i) for i in os.listdir(val_mask_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_mask_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()
    val_mask_path.sort()

    assert len(train_visible_path) == len(train_infrared_path),' The length of train dataset does not match. low:{}, high:{}'.\
                                         format(len(train_visible_path),len(train_infrared_path))
    assert len(val_visible_path) == len(val_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.\
                                          format(len(val_visible_path),len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path=train_visible_path[index]
        img_infrared_path=train_infrared_path[index]
        img_mask_path=train_mask_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)
        train_images_mask_path.append(img_mask_path)

    for index in range(len(val_visible_path)):
        img_visible_path=val_visible_path[index]
        img_infrared_path=val_infrared_path[index]
        img_mask_path=val_mask_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)
        val_images_mask_path.append(img_mask_path)

    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} mask images for training.".format(len(train_mask_path)))

    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.".format(len(val_infrared_path)))
    print("{} mask images for validation.\n".format(len(val_mask_path)))

    train_path_list = [(train_visible_path[i], train_infrared_path[i], train_mask_path[i]) for i in range(len(train_visible_path))]
    val_path_list = [(val_visible_path[i], val_infrared_path[i], val_mask_path[i]) for i in range(len(val_visible_path))]
    return train_path_list, val_path_list


def train_one_epoch(args,model,sam_model, optimizer, lr_scheduler, data_loader, device, epoch):
    utils.fix_randseed(args.seed + epoch)
    model.module.train_mode()
    sam_model.train()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    #accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    #accu_color_loss = torch.zeros(1).to(device)
    accu_grad_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    path="/root/autodl-tmp/experiments/"######################################################################
    for step, data in enumerate(data_loader):
        I_A_gt, I_B_gt, I_A, I_B, mask, name = data

        if torch.cuda.is_available():
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_A = I_A.to(device)####vis query
            I_B = I_B.to(device)####ir supp
            mask = mask.to(device)

        protos,_=model(args.condition, I_A, I_B, mask, name)
        I_fused =sam_model(I_A_gt, name, protos)
       
        I_fused= F.interpolate(I_fused, size=(512, 512), mode='bilinear', align_corners=True)
        fused_img = tensor2numpy(I_fused)##512,512,3
        save_pic(fused_img, "/root/autodl-tmp/experiments/train", str(name[0]))
        loss, loss_max, loss_grad = loss_function_prompt(I_A_gt, I_B_gt, I_fused, name)
        
        loss.backward()
        
        accu_total_loss += loss.detach()
        #accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        #accu_color_loss += loss_color.detach()
        accu_grad_loss += loss_grad.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  max loss: {:.3f}  grad loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_grad_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1),  accu_max_loss.item() / (step + 1),  accu_grad_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(args, model, sam_model, data_loader, device, epoch, lr, filefold_path):
    utils.fix_randseed(args.seed)
    loss_function_prompt = fusion_prompt_loss()

    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    #accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    #accu_color_loss = torch.zeros(1).to(device)
    accu_grad_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)
    
    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A_gt, I_B_gt, I_A, I_B, mask, name = data

        if torch.cuda.is_available():
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            mask = mask.to(device)
            

        protos, _=model(args.condition, I_A, I_B, mask, name)#######
        I_fused =sam_model(I_A_gt, name, protos)###
        I_fused= F.interpolate(I_fused, size=(512, 512), mode='bilinear', align_corners=True)
        if epoch % save_epoch == 0:
            if cnt <= save_length:
                img_vis_gt = tensor2numpy(I_A_gt)######
                img_ir_gt = tensor2numpy(I_B_gt)#######
                save_pic(img_vis_gt, evalfold_path, str(name[0]) + "vis")
                save_pic(img_ir_gt, evalfold_path, str(name[0]) + "ir")
                fused_img = tensor2numpy(I_fused)
                save_pic(fused_img, evalfold_path, str(name[0]))
                cnt += 1

        loss, loss_max, loss_grad = loss_function_prompt(I_A_gt, I_B_gt, I_fused, name)

        accu_total_loss += loss.detach()
        #accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        #accu_color_loss += loss_color
        accu_grad_loss += loss_grad.detach()

        data_loader.desc = "[val epoch {}] loss: {:.3f}  max loss: {:.3f}  grad loss: {:.3f}  lr: {:.6f}".format(epoch, accu_total_loss.item() / (step + 1),
            accu_max_loss.item() / (step + 1), accu_grad_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_grad_loss.item() / (step + 1)

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def save_pic(outputpic, path, index : str):
    #print(outputpic.shape)####3（512，512，3）
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)


def show_img(images,imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)

def tensor2numpy(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R

def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3

def unnormalize(img):
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]
    img = img.clone()
    for im_channel, mean, std in zip(img, mean_img, std_img):
        im_channel.mul_(std).add_(mean)
    return img

def to_numpy(tensor, type):
    if type== 'img':
        # Unnormalize the tensor
        unnormalized_tensor = unnormalize(tensor)
        
        # Ensure the tensor has the correct shape
        if unnormalized_tensor.dim() == 4:
            # Select the first image in the batch
            unnormalized_tensor = unnormalized_tensor[0]
        
        # Convert the tensor to a PIL image
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(unnormalized_tensor)
        
        # Convert the PIL image to a NumPy array
        image_array = np.array(pil_image).astype(np.uint8)
        
        return image_array
    elif type== 'mask':
        # Convert the mask tensor to a NumPy array
        mask_array = tensor.numpy().astype(np.uint8)
        
        # Ensure the mask has the correct shape
        if mask_array.ndim == 4:
            # Select the first mask in the batch
            mask_array = mask_array[0, 0]
        
        return mask_array
    else:
        raise ValueError("Unsupported type_ value. Use 'img' or 'mask'.")
