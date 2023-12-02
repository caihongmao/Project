import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os
import cv2
import utils.gaze as gaze_util
from utils.gazemap import *

class ColumbiaGaze(Dataset):

    def __init__(self, dir: str = './ColumbiaGazeDataSet/datasets_columbia/L'):
        self.dir = dir
        self.eval_entries = []

        # 加载数据集信息
        self._load_data_entries()

    def __len__(self):
        return len(self.eval_entries)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._load_sample(idx)

    def _load_data_entries(self):
        # 根据图像文件名解析数据集信息
        image_files = glob.glob(os.path.join(self.dir, '**', '*.jpg'), recursive=True)
        for image_file in image_files:
            entry = {
                'img_path': image_file, 
            }
            self.eval_entries.append(entry)

    def _load_sample(self, i):
        entry = self.eval_entries[i]

        img_path = entry['img_path']        
        array = img_path[:-4].split('_')
        gaze = []
        gaze.append(int(array[-2][:-1]))
        gaze.append(int(array[-1][:-1]))
        gaze =np.array(gaze) / 180 * np.pi

        # 从图像文件加载图像数据
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 对图像进行预处理
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_ = cv2.equalizeHist(img_)
        img_ = img_ / 255.
        img_ = img_ * 2 - 1
        img_ = img_.astype(np.float32)
        
        gmap = from_gaze2d(gaze.astype(np.float32))

        return {
            'eye': torch.tensor(img_.astype(np.float32)).unsqueeze(0),
            'gaze': torch.tensor(gaze.astype(np.float32)),
            'gmap': torch.tensor(gmap.astype(np.float32)),
            'img': img
        }