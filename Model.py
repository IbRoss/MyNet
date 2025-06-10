import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim


#CNN ------------------------------------------------------------ 
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(2, 2)  
        )

    def forward(self, x):
        return self.model(x)

class ImageDataset(Dataset):
    def __init__(self, hr_image_dir, lr_image_dir, transform=None, limit=115000):
        self.hr_image_dir = hr_image_dir
        self.lr_image_dir = lr_image_dir
        self.transform = transform
        
        self.image_names = sorted(os.listdir(hr_image_dir))[:limit]
        print(f"Number of images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        base_image_name = self.image_names[idx]
        
        hr_img_name = os.path.join(self.hr_image_dir, base_image_name)
        
        file_name, file_ext = os.path.splitext(base_image_name)
        
        lr_img_name = os.path.join(self.lr_image_dir, f"{file_name}_512{file_ext}")
        
        hr_image = Image.open(hr_img_name).convert("RGB")
        lr_image = Image.open(lr_img_name).convert("RGB")
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        
        return hr_image, lr_image

# Metrics functions----------------------------------------------
def mse(original, compared):
    err = np.mean((np.array(original) - np.array(compared)) ** 2)
    return err

def psnr(original, compared):
    mse_value = mse(original, compared)
    if mse_value == 0:
        return 100
    max_pixel = 255.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse_value))
    return psnr_value

def calculate_ssim(original, compared):
    original_array = np.array(original)
    compared_array = np.array(compared)
    if original_array.shape[-1] == 4:
        original_array = original_array[..., :3]
    if compared_array.shape[-1] == 4:
        compared_array = compared_array[..., :3]
    data_range = original_array.max() - original_array.min()  
    ssim_value, _ = ssim(original_array, compared_array, multichannel=True, full=True, win_size=3,data_range=data_range)
    return ssim_value

def angular_error(original, compared):
    original_array = np.array(original)
    compared_array = np.array(compared)
    if original_array.shape[-1] == 4:
        original_array = original_array[..., :3]
    if compared_array.shape[-1] == 4:
        compared_array = compared_array[..., :3]
    original_array = (original_array / 255.0) * 2.0 - 1.0
    compared_array = (compared_array / 255.0) * 2.0 - 1.0
    original_array = original_array / np.linalg.norm(original_array, axis=-1, keepdims=True)
    compared_array = compared_array / np.linalg.norm(compared_array, axis=-1, keepdims=True)
    dot_product = np.sum(original_array * compared_array, axis=-1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_error_radians = np.arccos(dot_product)
    angular_error_normalized = angular_error_radians / np.pi
    return np.mean(angular_error_normalized)

# Min Max normalization
mse_min, mse_max = 0.037689924, 62.54398608
psnr_min, psnr_max = 30.16894804, 62.36855096
ssim_min, ssim_max = 0.918334566, 0.9999956
angular_min, angular_max = 0.000138745, 0.07308881

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)



