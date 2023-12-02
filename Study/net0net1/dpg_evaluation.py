import torch
import numpy as np

from utils.filter import *
from utils.func import *
from utils.gaze import *
import torch
from torch.utils.data import DataLoader, random_split
from models.dpg import *
from datasources.unityeyes import *
from datasources.mpii_gaze import *

# GPU
import torch.backends.cudnn as cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

dpg_model = DPG()
dpg_model.load_state_dict(torch.load('models/dpg/model-dpg-0.1-epoch-39-loss-0.6479.pth'))
dpg_model.eval()
dpg_model = dpg_model.cuda()

val_root = "val"
val_dataset = UnityEyesDataset('datasets/val', eye_image_shape=(36, 60),
                                generate_heatmaps=True,
                                random_difficulty=True)
val_dataloader = DataLoader(val_dataset, batch_size=1)

total_angular_loss = 0.0
i = 0

while i <= 40000:
    entry = next(iter(val_dataloader))
    eye_input = entry['eye'].cuda()
    gaze_label = entry['gaze'].cuda()

    predict = dpg_model(eye_input)[0].reshape(1, 2).cpu().detach().numpy()
    label = gaze_label.cpu().detach().numpy()
    loss = np.mean(gaze_util.angular_error(predict, label))

    total_angular_loss += loss
    i += 1
    if i % 50 == 0:
        print(f'gaze_angular_loss_average_{i}_samples: ', total_angular_loss / (i))

print('gaze_angular_loss_average: ', total_angular_loss / 40000)
