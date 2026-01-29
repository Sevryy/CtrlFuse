from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F
import cv2
class FMBDataSet(Dataset):
    def __init__(self,image_path_list,phase="train"):
        self.phase = phase
        self.image_path_list = image_path_list
        self.transform = transforms.ToTensor()
        # Create a list to hold all sample indices
        #self.class_indices = list(range(len(self.image_path_list)))

    def __len__(self):
        return len(self.image_path_list)
 
    def __getitem__(self, item):
        visible_path, infrared_path, mask_path = self.image_path_list[item]

        image_visible_gt1 = Image.open(visible_path).convert('L')
        # 将单通道灰度图转换为三通道灰度图
        image_visible_gt = Image.new('RGB', image_visible_gt1.size)
        for x in range(image_visible_gt1.width):
            for y in range(image_visible_gt1.height):
                pixel_value = image_visible_gt1.getpixel((x, y))
                image_visible_gt.putpixel((x, y), (pixel_value, pixel_value, pixel_value))
        image_infrared_gt = Image.open(infrared_path).convert('RGB')           
        image_mask = Image.open(mask_path).convert('L')
         # Apply any specified transformations 
        if self.transform is not None:
            image_visible = self.transform(image_visible_gt)
            image_infrared = self.transform(image_infrared_gt)
            image_mask = self.transform(image_mask)
        name = visible_path.split(os.sep)[-1].split(".")[0]
        return image_visible, image_infrared, image_mask, name

    @staticmethod
    def collate_fn(batch):
        images_visible, images_infrared, images_mask, name = zip(*batch)
        images_visible = torch.stack(images_visible, dim=0)
        images_infrared = torch.stack(images_infrared, dim=0)
        images_mask = torch.stack(images_mask, dim=0)
        return  images_visible, images_infrared, images_mask, name