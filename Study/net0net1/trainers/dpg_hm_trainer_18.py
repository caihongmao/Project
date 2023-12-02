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

class dpg_hm_trainer_18(object):
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
        super(dpg_hm_trainer_18, self).__init__()

        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.current_learning_rate = initial_learning_rate
        self.loss_obj = torch.nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.current_learning_rate, weight_decay=1e-4)
        self.last_val_loss = math.inf
        self.lowest_val_loss = math.inf

        self.log = {
            'loss_train':[],
            'loss_val':[],
            'title':''
        }

        self.version = version

        step_size = 1  # 每隔多少个 epoch 调整一次学习率
        gamma = 0.95     # 学习率调整的比例
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

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
    
    def compute_coord_loss(self, predict, label):
        loss = self.loss_obj(predict, label)
        return loss
    
    def train_step(self, inputs):
        eye = inputs['eye'].to(self.device)
        gaze_label = inputs['gaze'].to(self.device)
        gmap_label = inputs['gmap'].to(self.device)
        heatmaps_label = inputs['heatmaps'].to(self.device)
        ldmks_label = inputs['landmarks'].to(self.device)
        radius_label = inputs['radius'].to(self.device)

        gaze_pred, gmap_pred, _, _, heatmaps_pred, ldmks_pred, radius_pred = self.model(eye)

        loss_gaze = self.model.dense_loss(gaze_pred, gaze_label)* self.batch_size
        loss_gmap = self.model.gmap_loss(gmap_pred, gmap_label) * 1e-5 * self.batch_size

        loss_heatmaps = self.compute_coord_loss(heatmaps_label, heatmaps_pred)
        loss_ldmks = self.compute_coord_loss(ldmks_pred, ldmks_label)
        loss_radius = self.compute_coord_loss(radius_pred, torch.unsqueeze(radius_label, dim=-1))

        # loss_heatmaps = torch.tensor(0)
        # loss_ldmks = torch.tensor(0)
        # loss_radius = torch.tensor(0)

        loss_angel = self.compute_angular_loss(gaze_pred, gaze_label)

        loss = 1000 * loss_heatmaps + loss_ldmks +  loss_radius
        # loss = loss_gmap + loss_gaze

        # 梯度清零
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_gmap.item(), loss_gaze.item(), loss_angel.item(), loss_heatmaps.item(), loss_ldmks.item(), loss_radius.item()

    def val_step(self, inputs):
        eye = inputs['eye'].to(self.device)
        gaze_label = inputs['gaze'].to(self.device)
        gmap_label = inputs['gmap'].to(self.device)
        heatmaps_label = inputs['heatmaps'].to(self.device)
        ldmks_label = inputs['landmarks'].to(self.device)
        radius_label = inputs['radius'].to(self.device)

        gaze_pred, gmap_pred, _, _, heatmaps_pred, ldmks_pred, radius_pred = self.model(eye)

        loss_gaze = self.model.dense_loss(gaze_pred, gaze_label)* self.batch_size
        loss_gmap = self.model.gmap_loss(gmap_pred, gmap_label) * 1e-5 * self.batch_size

        loss_heatmaps = self.compute_coord_loss(heatmaps_label, heatmaps_pred)
        loss_ldmks = self.compute_coord_loss(ldmks_pred, ldmks_label)
        loss_radius = self.compute_coord_loss(radius_pred, torch.unsqueeze(radius_label, dim=-1))

        # loss_heatmaps = torch.tensor(0)
        # loss_ldmks = torch.tensor(0)
        # loss_radius = torch.tensor(0)

        loss_angel = self.compute_angular_loss(gaze_pred, gaze_label)

        loss = 1000 * loss_heatmaps + loss_ldmks +  loss_radius
        # loss = loss_gmap + loss_gaze

        return loss.item(), loss_gmap.item(), loss_gaze.item(), loss_angel.item(), loss_heatmaps.item(), loss_ldmks.item(), loss_radius.item()

    def train_epoch(self, dataset):
        print('a new epoch training...')
        self.model.train()
        total_loss = 0.0
        total_loss_gaze = 0.0
        num_train_batches = 0.0

        for one_batch in dataset:
            batch_loss, loss_map, loss_gaze, loss_angel, loss_heatmaps, loss_ldmks, loss_radius = self.train_step(one_batch)
            total_loss += batch_loss
            total_loss_gaze += loss_gaze
            num_train_batches += 1

            self.log['loss_train'].append(batch_loss)
            loss_vis(self.log)

            if num_train_batches % 10 == 0:
                print('Trained batch:', num_train_batches,
                        'Batch loss:', batch_loss,
                        'Map loss', loss_map,
                        'Gaze loss', loss_gaze,
                        'Angel loss', loss_angel,
                        'Hm loss', loss_heatmaps,
                        'Ldmks loss', loss_ldmks,
                        'Radius loss', loss_radius,
                        'Epoch total loss:', total_loss)
                
            if num_train_batches >= 50000 / self.batch_size:
                break

        return total_loss / num_train_batches, total_loss_gaze / num_train_batches
    
    def val_epoch(self, dataset):
        print('a new epoch validating...')
        with torch.no_grad():
            self.model.eval()
            total_loss = 0.0
            total_loss_gaze = 0.0
            num_val_batches = 0.0
            for one_batch in dataset:
                batch_loss, loss_map, loss_gaze, loss_angel, loss_heatmaps, loss_ldmks, loss_radius = self.val_step(one_batch)
                total_loss += batch_loss
                total_loss_gaze += loss_gaze
                num_val_batches += 1

                if num_val_batches % 10 == 0:
                    print('Validated batch:', num_val_batches,
                            'Batch loss:', batch_loss,
                            'Map loss', loss_map,
                            'Gaze loss', loss_gaze,
                            'Angel loss', loss_angel,
                            'Hm loss', loss_heatmaps,
                            'Ldmks loss', loss_ldmks,
                            'Radius loss', loss_radius,
                            'Epoch total loss:', total_loss)
                    
                if num_val_batches >= 5000 / self.batch_size:
                    break

            # 长度对齐
            l = len(self.log['loss_train']) - len(self.log['loss_val'])
            if len(self.log['loss_val']) > 0:
                self.log['loss_val'] = self.log['loss_val'] + [self.log['loss_val'][-1] for i in range(l)]
            else:
                self.log['loss_val'] = [self.log['loss_train'][0] for i in range(l)]
            self.log['loss_val'][-1] = total_loss / num_val_batches
            loss_vis(self.log)
            return total_loss / num_val_batches,  total_loss_gaze / num_val_batches
    
    def run(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
        
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print('Start epoch {} with learning rate {}'.format(epoch, current_lr))

            train_loss, train_loss_gaze = self.train_epoch(self.train_dataloader)
            print('Epoch {} train batch loss {} train loss gaze {}'.format(epoch, train_loss / self.batch_size, train_loss_gaze / self.batch_size))

            val_loss, val_loss_gaze = self.val_epoch(self.val_dataloader)
            print('Epoch {} val batch loss {} val loss gaze {}'.format(epoch, val_loss / self.batch_size, val_loss_gaze / self.batch_size))

            # save model when reach a new lowest validation loss
            if val_loss < self.lowest_val_loss:
                if not os.path.exists(os.path.join('./models/dpghm18')):
                    os.makedirs(os.path.join('./models/dpghm18'))
                model_name = './models/dpghm18/model-dpghm18-{}-epoch-{}-loss-{:.4f}.pth'.format(self.version, epoch, val_loss_gaze)
                torch.save(self.model.state_dict(), model_name)
                plt.savefig('./models/dpghm18/model-dpghm18-{}-epoch-{}-loss-{:.4f}.png'.format(self.version, epoch, val_loss_gaze))
                print(f'Save model at: {model_name}')
                self.best_model = model_name
                self.lowest_val_loss = val_loss
            self.last_val_loss = val_loss

        return self.best_model