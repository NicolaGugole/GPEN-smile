import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, in_path, out_path, txt_file, resolution=512):
        self.resolution = resolution
        self.input_imgs = [os.path.join(in_path,i)[:-1] for i in open(txt_file) if 'jpg' in i]  # remove '\n' at the end
        self.output_imgs = [os.path.join(out_path,i)[:-1] for i in open(txt_file) if 'jpg' in i]

        self.length = len(self.input_imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_input = cv2.imread(self.input_imgs[index], cv2.IMREAD_COLOR)
        img_input = cv2.resize(img_input, (self.resolution, self.resolution))
        
        img_input = img_input.astype(np.float32)/255.


        img_output = cv2.imread(self.output_imgs[index], cv2.IMREAD_COLOR)
        img_output = cv2.resize(img_output, (self.resolution, self.resolution))
        
        img_output = img_output.astype(np.float32)/255.

        img_output =  (torch.from_numpy(img_output) - 0.5) / 0.5   # map between -1, 1
        img_input =  (torch.from_numpy(img_input) - 0.5) / 0.5
        
        img_output = img_output.permute(2, 0, 1).flip(0) # BGR->RGB
        img_input = img_input.permute(2, 0, 1).flip(0) # BGR->RGB

        return img_input, img_output

