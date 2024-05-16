import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors as tv
import matplotlib.pyplot as plt
import os
from utils.visualization import plot_tensors as plot
from utils.custom_transforms import InnerRandomCrop, Resize


class SemSegDataset(Dataset):
    def __init__(self, file_list, base_path=None, transform=None, max_classes=81, mode="train_val", resize=(256, 256)):
        self.file_list = file_list
        if transform is None:
            self.use_default_transform = True
        self.base_path = base_path
        self.max_classes = max_classes - 1 #remove background class

        assert mode in ["train_val", "test"], "Mode should be either 'train_val' or 'test'"
        self.mode = mode
        self.resize = resize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx].split()
        # Read image

        if self.base_path:
            img_path = os.path.join(self.base_path, img_path)
            mask_path = os.path.join(self.base_path, mask_path)

        # read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found: {img_path}")
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Mask not found: {mask_path}")
            return None, None


        mask = tv.Mask(mask)
        mask[mask>self.max_classes] = 0

        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if not self.use_default_transform:
            t_img, t_mask = self.transform(img, mask)
        else:
            smaller_dim = min(img.shape[1:])
            if self.mode == "train_val":
                self.transform = transforms.Compose(
                    [
                        InnerRandomCrop(smaller_dim, smaller_dim),
                        Resize(self.resize[0], self.resize[1]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.CenterCrop(smaller_dim),
                    Resize(self.resize[0], self.resize[1]),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            t_img, t_mask = self.transform(img, mask)


        #visualize for debug
        # plot([[img, t_img],[mask, t_mask]], row_title=["Image", "Mask"])
        # plt.show()

        return t_img, t_mask