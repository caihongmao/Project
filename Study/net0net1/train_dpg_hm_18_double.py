from models.dpg_hm_18_double import *
from models.dpg import *
from utils.gazemap import *
from trainers.dpg_hm_trainer_18_double import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from datasources.unityeyes import *
from datasources.columbia_gaze_double import *

# GPU
import torch.backends.cudnn as cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 128

# 原始模型
net =  DPGHM18Double()
net.load_state_dict(torch.load('models/dpghm18double/init_dpg_hm18_double.pth'))

for param in net.DPGHM18.parameters():
    param.requires_grad = False


# 数据
# train_dataset = UnityEyesDataset('datasets/train', eye_image_shape=(36, 60), random_difficulty=True) 
# val_dataset = UnityEyesDataset('datasets/val', eye_image_shape=(36, 60), random_difficulty=True)

dataset = ColumbiaGaze()
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% 用于训练
val_size = dataset_size - train_size  # 剩余部分用于验证
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(train_dataset, 
                                pin_memory=True, 
                                drop_last=True,
                                # shuffle=True,
                                batch_size=batch_size)

val_dataloader = DataLoader(val_dataset, 
                                pin_memory=True, 
                                drop_last=False,
                                batch_size=batch_size)

# 训练
trainer = dpg_hm_trainer_18(net, train_dataloader, val_dataloader, 
                     device=device, 
                     batch_size=batch_size, 
                     version = 'doubel_linear',
                     initial_learning_rate = 1e-3,
                     epochs=200, 
                     start_epoch=1)

trainer.run()