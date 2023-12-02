import torch
import numpy as np

from utils.filter import *
from utils.func import *
from utils.gaze import *
import torch
from torch.utils.data import DataLoader, random_split
from models.elg import *
from datasources.unityeyes import *
from datasources.mpii_gaze import *

# GPU
import torch.backends.cudnn as cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

elg_model = ELG()
elg_model.load_state_dict(torch.load('models/elg/model-elg-0.3-epoch-27-loss-0.7677.pth'))
elg_model = torch.load('models/checkpoints/model-v0.2-(36, 60)-epoch-89-loss-0.7151.pth')
elg_model.eval()
elg_model = elg_model.cuda()

# 回归网络
fc_model = torch.load('models/checkpoints/model-Gaze-v0.2-Gaze-epoch-16-loss-16.5713.pth')
fc_model.eval()

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

    landmarks_label = entry['landmarks'].cuda()
    radius_label = entry['radius'].cuda()
    gaze_label = entry['gaze'].cuda()

    heatmaps_predict, ldmks_predict, radius_predict = elg_model(eye_input)

    # 解析视角
    predict = fc_model(ldmks_predict).reshape(1, 2).cpu().detach().numpy()
    label = gaze_label.cpu().detach().numpy()
    loss = np.mean(gaze_util.angular_error(predict, label))

    total_angular_loss += loss
    i += 1
    if i % 50 == 0:
        print(f'gaze_angular_loss_average_{i}_samples: ', total_angular_loss / (i))

print('gaze_angular_loss_average: ', total_angular_loss / 40000)
