import torch
from torch import nn
import random
from tqdm.notebook import tqdm
import numpy as np

class Trainer(object):
    def __init__(self, model, optim=None, scheduler=None, 
                    train_loader=None, val_loader=None, test_loader=None, device=None):
        self.loss_ftn = nn.CrossEntropyLoss()
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def test_acc(self, cmd=None):
        if cmd == 'val':
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        val_loss = 0
        correct, total = 0, 0
        self.model.eval()
        with torch.no_grad():
            for _, (images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_ftn(outputs, labels)
                val_loss += loss.item()*images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return val_loss / len(data_loader.dataset), 100 * correct / total


    def train(self, n_epoch, verbose_freq=50):
        for i in tqdm(range(n_epoch)):
            self.model.train()
            train_loss = 0
            correct, total = 0, 0
            for _, (images, labels) in enumerate(self.train_loader):
                # If gpu is available, add data to cuda
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optim.zero_grad()
                outputs = self.model(images)
                loss = self.loss_ftn(outputs, labels)
                loss.backward()
                self.optim.step()
                train_loss += loss.item()*images.size(0)

                # Caculate train accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            if (i+1)%verbose_freq == 0:            
                val_loss, val_acc = self.test_acc(cmd='val')
                print('===> Epoch: {} / Avg train loss: {:.3f} / Train acc: {:.3f}% / Avg val loss: {:.3f} / Val acc: {:.3f}%'.format(
                    i+1, train_loss / len(self.train_loader.dataset), 100 * correct / total, val_loss, val_acc))
            
            # Update learning rates
            self.scheduler.step()


# Trainer class to implement CutMix & MixUp
class AugTrainer(Trainer):
    def __init__(self, model, optim=None, scheduler=None, 
                    train_loader=None, val_loader=None, test_loader=None, device=None):
        self.loss_ftn = nn.CrossEntropyLoss()
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

    def mixup_data(self, x, y):
        lam = np.random.beta(1, 1)
        index = torch.randperm(x.size()[0]).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def rand_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        # uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, x, y):
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(x.size()[0]).to(self.device)

        labels_a, labels_b = y, y[rand_index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size()[2], x.size()[3], lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        return x, labels_a, labels_b, lam

    def train(self, n_epoch, verbose_freq=20, threshold=0.5):
        for i in tqdm(range(n_epoch)):
            self.model.train()
            train_loss = 0
            correct, total = 0, 0
            
            for _, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # CutMix
                if random.random() < threshold:
                    images, labels_a, labels_b, lam = self.cutmix_data(images, labels)
                # MixUp
                else:
                    images, labels_a, labels_b, lam = self.mixup_data(images, labels)
                
                # Update 
                self.optim.zero_grad()
                outputs = self.model(images)
                loss = lam * self.loss_ftn(outputs, labels_a) + (1 - lam) * self.loss_ftn(outputs, labels_b)
                loss.backward()
                self.optim.step()

                train_loss += loss.item()*images.size(0)
                # Calculate train accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (lam * predicted.eq(labels_a.data).sum().float() + 
                            (1 - lam) * predicted.eq(labels_b.data).sum().float())
            
            # Calculate train loss / validation loss / validation accuracy
            train_loss = train_loss / len(self.train_loader.dataset)
            val_loss, val_acc = self.test_acc(cmd='val')

            # Save train loss / validation loss / validation accuracy to tensorboard
            writer.add_scalar("Train Loss", train_loss, i+1)
            writer.add_scalar("Validation Loss", val_loss, i+1)
            writer.add_scalar("Validation Accuracy", val_acc, i+1)

            if (i+1) % verbose_freq == 0:           
                print('===> Epoch: {} / Avg train loss: {:.3f} / Train acc: {:.3f}% / Avg val loss: {:.3f} / Val acc: {:.3f}%'.format(
                    i+1, train_loss, 100 * correct / total, val_loss, val_acc))
            
            # Update learning rates
            self.scheduler.step()