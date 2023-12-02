import os
import torch
import torch.optim as optim
import math
from utils.visualize import *
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

# GPU
import torch.backends.cudnn as cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

class elg_gaze_trainer(object):
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 device,
                 version = '0.1',
                 batch_size=64,
                 epochs=100,
                 initial_learning_rate=1e-4,
                 start_epoch=1):
        super(elg_gaze_trainer, self).__init__()

        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.current_learning_rate = initial_learning_rate

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf

        self.loss_obj = torch.nn.MSELoss()

        self.log = {
            'loss_train':[],
            'loss_val':[],
            'title':''
        }

        self.version = version

        step_size = 1  # 每隔多少个 epoch 调整一次学习率
        gamma = 0.95     # 学习率调整的比例
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    
    def compute_coord_loss(self, predict, label):
        loss = self.loss_obj(predict, label)
        return loss
    
    def compute_angular_loss(self, predict, label):
        """Pytorch method to calculate angular loss (via cosine similarity)"""
        def angle_to_unit_vectors(y):
            sin = torch.sin(y)
            cos = torch.cos(y)
            return torch.stack([
                cos[:, 0] * sin[:, 1],
                sin[:, 0],
                cos[:, 0] * cos[:, 1],
            ], dim=1)

        a = angle_to_unit_vectors(predict)
        b = angle_to_unit_vectors(label)
        ab = torch.sum(a*b, dim=1)
        a_norm = torch.sqrt(torch.sum(torch.square(a), dim=1))
        b_norm = torch.sqrt(torch.sum(torch.square(b), dim=1))
        cos_sim = ab / (a_norm * b_norm)
        cos_sim = torch.clip(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
        ang = torch.acos(cos_sim) * 180. / math.pi
        return torch.mean(ang)
    
    def train_step(self, inputs):
        ldmks_input = inputs['landmarks'].to(self.device)
        gaze_label = inputs['gaze'].to(self.device)
        gaze_predict = self.model(ldmks_input)

        loss_angle = self.compute_angular_loss(gaze_predict, gaze_label) 
        loss = self.compute_coord_loss(gaze_predict, gaze_label) 
       
        # 梯度清零
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_angle.item()

    def val_step(self, inputs):
        ldmks_input = inputs['landmarks'].to(self.device)
        gaze_label = inputs['gaze'].to(self.device)
        gaze_predict = self.model(ldmks_input)

        loss_angle = self.compute_angular_loss(gaze_predict, gaze_label)
        loss = self.compute_coord_loss(gaze_predict, gaze_label) 
    
        return loss.item(), loss_angle.item()

    def train_epoch(self, dataset):
        print('a new epoch training...')
        self.model.train()

        total_loss = 0.0
        total_loss_angel = 0.0
        num_train_batches = 0.0

        for one_batch in dataset:
            gaze_loss, loss_angel = self.train_step(one_batch)
            total_loss += gaze_loss
            total_loss_angel += loss_angel
            num_train_batches += 1

            self.log['loss_train'].append(gaze_loss)
            loss_vis(self.log)

            if num_train_batches % 10 == 0:
                print('Trained batch:', num_train_batches,
                        'Gaze loss:', gaze_loss,
                        'Angel loss', loss_angel,
                        'Epoch total loss:', total_loss)
            if num_train_batches >= 15000 / self.batch_size:
                break
        return total_loss / num_train_batches, total_loss_angel / num_train_batches
    
    def val_epoch(self, dataset):
        print('a new epoch validating...')
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.0
            total_loss_angel = 0.0
            num_val_batches = 0.0

            for one_batch in dataset:
                gaze_loss, loss_angel = self.val_step(one_batch)
                total_loss += gaze_loss
                total_loss_angel += loss_angel
                num_val_batches += 1

                if num_val_batches % 10 == 0:
                    print('Validated batch:', num_val_batches,
                        'Gaze loss:', gaze_loss,
                        'Angel loss', loss_angel,
                        'Epoch total loss:', total_loss)
                if num_val_batches >= 3000 / self.batch_size:
                    break

            # 长度对齐
            l = len(self.log['loss_train']) - len(self.log['loss_val'])
            if len(self.log['loss_val']) > 0:
                self.log['loss_val'] = self.log['loss_val'] + [self.log['loss_val'][-1] for i in range(l)]
            else:
                self.log['loss_val'] = [self.log['loss_train'][0] for i in range(l)]
            self.log['loss_val'][-1] = total_loss / num_val_batches
            loss_vis(self.log)
            return total_loss / num_val_batches, total_loss_angel / num_val_batches
    
    def run(self):
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.start_epoch, self.epochs + 1):
        
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print('Start epoch {} with learning rate {}'.format(epoch, current_lr))

            train_loss, train_loss_angel = self.train_epoch(self.train_dataloader)
            print('Epoch {} train gaze loss {} train angel loss {}'.format(epoch, train_loss / self.batch_size, train_loss_angel / self.batch_size))

            val_loss, val_loss_angel = self.val_epoch(self.val_dataloader)
            print('Epoch {} val gaze loss {} val angel loss {}'.format(epoch, val_loss / self.batch_size, val_loss_angel / self.batch_size))

            # save model when reach a new lowest validation loss
            if val_loss < self.lowest_val_loss:
                if not os.path.exists(os.path.join('./models/gaze')):
                    os.makedirs(os.path.join('./models/gaze'))
                model_name = './models/gaze/model-gaze-{}-epoch-{}-loss-{:.4f}.pth'.format(self.version, epoch, val_loss_angel)
                torch.save(self.model.state_dict(), model_name)
                plt.savefig('./models/gaze/model-gaze-{}-epoch-{}-loss-{:.4f}.png'.format(self.version, epoch, val_loss_angel))
                print(f'Save model at: {model_name}')
                self.best_model = model_name
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss

        return self.best_model