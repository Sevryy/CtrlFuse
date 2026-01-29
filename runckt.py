import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
import torch
import os
import cv2
import statistics
import warnings
from torchvision import transforms
from dataload.FMB import FMBDataSet
from scripts.utils import read_data
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter
from torch.utils.data import DataLoader
from dataload.FMB import FMBDataSet
from torchvision import transforms
from Net import net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")
f_dir = '/root/autodl-tmp/experiments/ex14.4_FMB/gray'
s_dir = '/root/autodl-tmp/experiments/ex14.4_FMB/seg'
if not os.path.exists(f_dir): os.makedirs(f_dir,exist_ok=True)
if not os.path.exists(s_dir): os.makedirs(s_dir,exist_ok=True)
model_weight_path='/root/autodl-tmp/CtrlFuse/pth/checkpoint_best.pth'
test_root = '/root/pro/FMB/test'
test_images_visible_path = []
test_images_infrared_path = []
test_images_mask_path=[]
supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  
test_visible_root = os.path.join(test_root, "vi")###########
test_infrared_root = os.path.join(test_root, "ir")##########
test_mask_root= os.path.join(test_root, "mask")#############
test_visible_path = [os.path.join(test_visible_root, i) for i in os.listdir(test_visible_root)
                  if os.path.splitext(i)[-1] in supported]
test_infrared_path = [os.path.join(test_infrared_root, i) for i in os.listdir(test_infrared_root)
                  if os.path.splitext(i)[-1] in supported]
test_mask_path = [os.path.join(test_mask_root, i) for i in os.listdir(test_mask_root)
                  if os.path.splitext(i)[-1] in supported]
test_visible_path.sort()
test_infrared_path.sort()
test_mask_path.sort()
assert len(test_visible_path) == len(test_infrared_path),' The length of val dataset does not match. low:{}, high:{}'.\
                                          format(len(test_visible_path),len(test_infrared_path))
print("Visible and Infrared images check finish")
for index in range(len(test_visible_path)):
        img_visible_path=test_visible_path[index]
        img_infrared_path=test_infrared_path[index]
        img_mask_path=test_mask_path[index]
        test_images_visible_path.append(img_visible_path)
        test_images_infrared_path.append(img_infrared_path)
        test_images_mask_path.append(img_mask_path)
print("{} visible images for test.".format(len(test_visible_path)))
print("{} infrared images for test.".format(len(test_infrared_path)))
print("{} mask images for test.\n".format(len(test_mask_path)))
test_path_list = [(test_visible_path[i], test_infrared_path[i], test_mask_path[i]) for i in range(len(test_visible_path))]
test_dataset=FMBDataSet(test_path_list)
testloader = DataLoader(test_dataset,
                            batch_size=1,
                            pin_memory=True,
                            num_workers=0,
                            collate_fn=test_dataset.collate_fn)
with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = net().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()
    for i, data in enumerate(testloader):
        I_A, I_B, mask, name = data
        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            mask = mask.to(device)
        I_ref,logit_mask_v,logit_mask_i,pred_mask_v,pred_mask_i,pred_mask,I_fused = model('mask', I_A, I_B, mask, name)
        fi = np.squeeze(I_fused.detach().cpu().numpy()) *255
        plt.imsave(os.path.join(f_dir, str(name[0]) + '.png'), fi, cmap="gray")
        seg_pred_mask=np.squeeze(pred_mask.detach().cpu().numpy()) * 255
        plt.imsave(os.path.join(s_dir, str(name[0]) + 'predmask.png'), seg_pred_mask, cmap="gray")


