from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import os
import cv2
import scipy.io as sio
import utils.gaze as gaze_util
from utils.gazemap import *


class MPIIGaze(Dataset):

    def __init__(self, mpii_dir: str = './MPIIGaze'):

        self.mpii_dir = mpii_dir
        eval_files = glob.glob(f'{mpii_dir}/Evaluation Subset/sample list for eye image/*.txt')

        self.eval_entries = []
        for ef in eval_files:
            person = os.path.splitext(os.path.basename(ef))[0]
            with open(ef) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line != '':
                        img_path, side = [x.strip() for x in line.split()]
                        day, img = img_path.split('/')
                        self.eval_entries.append({
                            'day': day,
                            'img_name': img,
                            'person': person,
                            'side': side
                        })

    def __len__(self):
        return len(self.eval_entries)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._load_sample(idx)

    def _load_sample(self, i):
        entry = self.eval_entries[i]
        mat_path = os.path.join(self.mpii_dir, 'Data/Normalized', entry['person'], entry['day'] + '.mat')
        mat = sio.loadmat(mat_path)

        filenames = mat['filenames']
        row = np.argwhere(filenames == entry['img_name'])[0][0]
        side = entry['side']

        img = mat['data'][side][0, 0]['image'][0, 0][row]
        img = cv2.equalizeHist(img)
        img_ = img / 255.
        img_ = img_ * 2 - 1
        img_ = img_.astype(np.float32)


        (x, y, z) = mat['data'][side][0, 0]['gaze'][0, 0][row]

        
        # theta = np.arcsin(-y)
        # phi = np.arctan2(-x, -z)
        # gaze = np.array([-theta, phi])

        gaze = gaze_util.vector_to_pitchyaw(np.array([-x, -y, -z]).reshape((1, 3))).flatten()

        # if side == 'right':
        #     img_ = np.flip(img_, axis=1)
        #     gaze[1] = -gaze[1]

        gmap =  from_gaze2d(gaze.astype(np.float32))
        return {
            'img':img,
            'vector':np.array([-x, -y, -z]),
            'eye': torch.tensor(img_.astype(np.float32)).unsqueeze(0),
            'gaze': torch.tensor(gaze.astype(np.float32)),
            'side': side,
            'gmap': torch.tensor(gmap.astype(np.float32)),
        }