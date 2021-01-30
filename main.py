import psutil
def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
print("="*40, "Memory Information", "="*40)
svmem = psutil.virtual_memory()
print(f"Total: {get_size(svmem.total)}") ; print(f"Available: {get_size(svmem.available)}")
print(f"Used: {get_size(svmem.used)}") ; print(f"Percentage: {svmem.percent}%")
! nvidia-smi


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import sys
import matplotlib.pyplot as plt
import imageio
from google.colab import drive 
from tensorflow import summary
import datetime
import pickle
from livelossplot import PlotLosses



%cd /content/drive/MyDrive/
%reload_ext tensorboard
current_time = str(datetime.datetime.now().timestamp())
train_log_dir = 'logs/tensorboard/train/'+current_time   
#test_log_dir = 'storage/test/' + current_time  
train_summary_writer = summary.create_file_writer(train_log_dir)
#test_summary_writer = summary.create_file_writer(test_log_dir)

%tensorboard --logdir logs/tensorboard

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()  # вызов init у родителя Generator = Module
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU())
        
        # 16 Residual blocks
        self.layer2 = Residual_block(64)
        self.layer3 = Residual_block(64)
        self.layer4 = Residual_block(64)
        self.layer5 = Residual_block(64)
        self.layer6 = Residual_block(64)
        self.layer7 = Residual_block(64)
        self.layer8 = Residual_block(64)
        self.layer9 = Residual_block(64)
        self.layer10 = Residual_block(64)
        self.layer11 = Residual_block(64)
        self.layer12 = Residual_block(64)
        self.layer13 = Residual_block(64)
        self.layer14 = Residual_block(64)
        self.layer15 = Residual_block(64)
        self.layer16 = Residual_block(64)
        self.layer17 = Residual_block(64)


        self.layer18 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.layer19 = Repiet_block(64)
        self.layer20 = Repiet_block(64)
        self.layer21 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.layer22 = nn.Tanh() #nn.Sigmoid() тк нужны значения [0;1]

    def forward(self, a):
        c = self.layer1(a)
        b = self.layer2(c)
        b = self.layer3(b)
        b = self.layer4(b)
        b = self.layer5(b)
        b = self.layer6(b)
        b = self.layer7(b)
        b = self.layer8(b)
        b = self.layer9(b)
        b = self.layer10(b)
        b = self.layer11(b)
        b = self.layer12(b)
        b = self.layer13(b)
        b = self.layer14(b)
        b = self.layer15(b)
        b = self.layer16(b)
        b = self.layer17(b)
        b = self.layer18(b)

        b = self.layer19(b + c)
        b = self.layer20(b)
        b = self.layer21(b)
        b = self.layer22(b)
        return (b + 1) / 2


class Residual_block(nn.Module):
    def __init__(self, numb):
        super(Residual_block, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=numb, out_channels=numb, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(numb)
        self.layer3 = nn.PReLU()
        self.layer4 = nn.Conv2d(in_channels=numb, out_channels=numb, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(numb)

    def forward(self, a):
        b = self.layer1(a)
        b = self.layer2(b)
        b = self.layer3(b)
        b = self.layer4(b)
        b = self.layer5(b)

        return a + b


class Repiet_block(nn.Module):
    def __init__(self, numb):
        super(Repiet_block, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=numb, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.PixelShuffle(upscale_factor=2)
        self.layer3 = nn.PReLU()

    def forward(self, a):
        b = self.layer1(a)
        b = self.layer2(b)
        b = self.layer3(b)

        return b


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer9 = nn.Sequential(
            nn.Linear(6*6*512, 1024),# bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1) #bias=True)
            #nn.Sigmoid()
        )

    def forward(self, a):
        b = self.layer1(a)
        b = self.layer2(b)
        b = self.layer3(b)
        b = self.layer4(b)
        b = self.layer5(b)
        b = self.layer6(b)
        b = self.layer7(b)
        b = self.layer8(b)
        b = b.view(b.size(0), -1) 
        b = self.layer9(b)
        ba = b.size(0) #  b.shape[0]
        b = torch.sigmoid(b.view(ba)) 

        return b


class Dataset(Dataset):  
    def __init__(self, direct, crop, resolution):
        super(Dataset, self).__init__()
        self.dataset = [os.path.join(direct, x) for x in os.listdir(direct)]  
        self.lr_transform = self.LR_crop(crop, resolution)
        self.hr_transform = self.HR_crop(crop)

    def __getitem__(self, index):
        path = self.dataset[index]
        hr_img = self.hr_transform(Image.open(path).convert("RGB"))
        lr_img = self.lr_transform(hr_img)  # [0; 1]

        hr_img = hr_img * 2 - 1  # [-1; 1]  

        return lr_img, hr_img

    def __len__(self):
        return len(self.dataset)

    def LR_crop(self, crop, interpolation_scale):
        data_transform_lr = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((crop//interpolation_scale, crop//interpolation_scale), \
                                                                  interpolation=Image.BICUBIC),
                                                transforms.ToTensor()]) 
        return data_transform_lr

    def HR_crop(self, crop):
        data_transform_hr = transforms.Compose([transforms.RandomCrop(crop),  
                                                transforms.Resize(crop), 
                                                transforms.ToTensor()])
        return data_transform_hr


class Loss_Generator(nn.Module):
    def __init__(self):
        super(Loss_Generator, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.mse_loss = nn.MSELoss()
        self.bceloss = nn.BCELoss()

        blocks = []
        blocks.append(vgg19.features[:36].eval())
        '''blocks.append(vgg19.features[:4].eval())
        blocks.append(vgg19.features[4:9].eval())
        blocks.append(vgg19.features[9:18].eval())
        blocks.append(vgg19.features[18:27].eval())
        blocks.append(vgg19.features[27:36].eval())'''

        blocks = nn.ModuleList(blocks)

        for bl in blocks.parameters():
            bl.requires_grad = False
        self.blocks = blocks
        '''mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)  для нормализации в imagenet
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)'''

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, hr_, sr_, out_discriminator):
        content_loss = 0.0
        output_x, output_y = [], []

        x = (hr_ + 1)/2
        y = (sr_ + 1)/2
        #x = (x - self.mean) / self.std 
        #y = (y - self.mean) / self.std
        #x = (hr_img- self.mean) / self.std  # ПОКА ЗАКОМЕНТИЛА
        #y = (sr_img - self.mean) / self.std
        for block in self.blocks:
            x = block(x).div(12.75)
            y = block(y).div(12.75)
            '''x = block(hr_img).div(12.75)
            y = block(sr_img).div(12.75)'''
            content_loss += self.mse_loss(y,x)

        ones = torch.ones_like(out_discriminator)
        ones = ones - 0.1
        adversarial_loss = self.bceloss(out_discriminator, ones) #torch.ones_like(out_discriminator))
        beta = 0.001

        return content_loss + beta * adversarial_loss


class Loss_Discriminator(nn.Module):
    def __init__(self):
        super(Loss_Discriminator, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, real, fake):
        one = torch.ones_like(real)
        one = one - 0.1
        zeros = torch.zeros_like(fake)
        zeros = zeros + 0.1

        real_loss = self.loss(real, one) #torch.ones_like(real))
        fake_loss = self.loss(fake, zeros) #torch.zeros_like(fake))

        return fake_loss + real_loss


def visualize_image(image, save_path, save_name):
    image = image.cpu().detach().numpy()
    image = image.transpose(1, 2, 0)
    plt.imsave(arr=image, fname='{}{}'.format(save_path, save_name))


use_gpu = torch.cuda.is_available()  # использование gpu
device = torch.device('cuda:0' if use_gpu else 'cpu')
print('use_gpu: ', use_gpu)


batch_size = 16
learning_rateD = 0.00001 #0.000001 
learning_rateG = 0.0001 #0.00001 
num_epoch = 10000


root = '/content/drive/MyDrive/data_download/dataset_images/test_256' #div2k' 
crop_size = 96
scale = 4

ep = []
k = 0  # шаги в графике лосса

train_dataset = Dataset(root, crop_size, scale)
train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


net_G = Generator()
net_D = Discriminator()

if use_gpu:  # использоване gpu
    net_G.to(device)
    net_D.to(device)

criterion_G = Loss_Generator().to(device) if use_gpu else Loss_Generator()
criterion_D = Loss_Discriminator().to(device) if use_gpu else Loss_Discriminator()
optimizer_G = optim.Adam(net_G.parameters(), lr=learning_rateG) 
optimizer_D = optim.Adam(net_D.parameters(), lr=learning_rateD) 

out_loss = open("/content/drive/MyDrive/history_loss.txt", "w")  # запись лоссов в файл

picture = 0

los_sum = 0.0
loss_g = 0.0
step_d = 0

for epoch in range(num_epoch):
    for i, (lr_img, hr_img) in enumerate(train):

        lr_img = Variable(lr_img, requires_grad=True).to(device) if use_gpu else Variable(lr_img, requires_grad=True)  #Variable(torch.tensor(lr_img, requires_grad=True)).
        hr_img = Variable(hr_img, requires_grad=True).to(device) if use_gpu else Variable(hr_img, requires_grad=True)  # Variable(torch.tensor(hr_img, requires_grad=True))
        
        # TRAIN DISCRIMINATOR
        if los_sum/(step_d + 1) >= loss_g:
          net_D.zero_grad()
           
        sr_img = net_G(lr_img)

        if los_sum/(step_d + 1) >= loss_g:
          real_out = net_D(hr_img)
          fake_out = net_D(sr_img)
           

        if los_sum/(step_d + 1) >= loss_g:
          loss_d = criterion_D(real_out, fake_out)
          loss_d.backward(retain_graph=True)
          optimizer_D.step()  
          los_sum += float(loss_d)
          step_d += 1     

        sr_img = sr_img*2 - 1

        # TRAIN GENERATOR
        net_G.zero_grad()
        fake_out = net_D(sr_img)
        loss_g = criterion_G(hr_img, sr_img, fake_out)
          
        loss_g.backward()#retain_graph=True)
        optimizer_G.step()  

        if (epoch+1) == 100:
          lrD = 0.000001
          lrG = 0.00001
          optimizer_G = optim.Adam(net_G.parameters(), lr=lrG)
          optimizer_D = optim.Adam(net_D.parameters(), lr=lrD)
         
           
        if (i+1) % 50 ==0:
          out_loss.write('{0} {1}\n'.format(loss_d.item(), loss_g.item()))

          # TENSORBOARD
          with train_summary_writer.as_default():
              summary.scalar('loss_d', loss_d.item(), step=k)
              summary.scalar('loss_g', loss_g.item(), step=k)
          k += 1
          
          print('Epoch [{}/{}], Step [{}], Loss_D: {:.5f}, Loss_G: {:.5f}'.format(epoch + 1, num_epoch, i + 1, loss_d.item(), loss_g.item()))
        
        # сохранение картинок
        if (i + 1) % 20 == 0:  # % 2
          torch.save(net_D.state_dict(), '/content/drive/MyDrive/data_download/dataset_images/modelD_3.pth')
          torch.save(net_G.state_dict(), '/content/drive/MyDrive/data_download/dataset_images/modelG_3.pth')

        if (i + 1) % 103 == 0:
          torch.save(net_D.state_dict(), '/content/drive/MyDrive/data_download/dataset_images/modelD_4.pth')
          torch.save(net_G.state_dict(), '/content/drive/MyDrive/data_download/dataset_images/modelG_4.pth')
        
        # сохранение картинок
        if (epoch + 1) % 20 == 0:
          for j in range(1):
            sr_img = (sr_img + 1) / 2
            save_path = '/content/drive/MyDrive/data_out/' 
            save_name = 'img-{}.jpg'.format(picture)
            picture += 1
            visualize_image(image=sr_img[j], save_path=save_path, save_name=save_name)
       

