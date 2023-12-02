from models.dpg_hm import *
from models.dpg import *
from utils.gazemap import *
from trainers.dpg_hm_trainer import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from datasources.unityeyes import *
from datasources.mpii_gaze import *

# GPU
import torch.backends.cudnn as cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 128

# 原始模型
net = DPGHM()
net.load_state_dict(torch.load('models/dpghm/model-dpghm-0.3-epoch-19-loss-0.6595.pth'))

for param in net.Hourglass_net.parameters():
    param.requires_grad = False
for param in net.DenseNet.parameters():
    param.requires_grad = False

# self.Hourglass_module = Hourglass_module(64, 64, 2)
# self.conv2 = conv1x1(64, 1)
# self.bn2 = nn.BatchNorm2d(1)
# self.relu2= nn.ReLU(inplace=True)

# 冻结
for param in net.HM.Hourglass_module.parameters():
    param.requires_grad = True
for param in net.HM.conv2.parameters():
    param.requires_grad = True
for param in net.HM.bn2.parameters():
    param.requires_grad = True
for param in net.HM.relu2.parameters():
    param.requires_grad = True

# 数据
train_dataset = UnityEyesDataset('datasets/train', eye_image_shape=(36, 60), random_difficulty=True) 
val_dataset = UnityEyesDataset('datasets/val', eye_image_shape=(36, 60), random_difficulty=True)

# dataset = MPIIGaze()
# dataset_size = len(dataset)
# train_size = int(0.8 * dataset_size)  # 80% 用于训练
# val_size = dataset_size - train_size  # 剩余部分用于验证
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

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
trainer = dpg_hm_trainer(net, train_dataloader, val_dataloader, 
                     device=device, 
                     batch_size=batch_size, 
                     version = '0.3',
                     initial_learning_rate = 1e-4,
                     epochs=200, 
                     start_epoch=1)

trainer.run()