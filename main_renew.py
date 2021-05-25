import psutil


def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


print("=" * 40, "Memory Information", "=" * 40)
svmem = psutil.virtual_memory()
print(f"Total: {get_size(svmem.total)}");
print(f"Available: {get_size(svmem.available)}")
print(f"Used: {get_size(svmem.used)}");
print(f"Percentage: {svmem.percent}%")
! nvidia - smi

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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
from tqdm import tqdm
from skimage.measure import compare_ssim, compare_psnr


class Generator(nn.Module):
  DEFAULT_RELU_LEAKINESS = 0.1

  def __init__(self, num_inputs, num_outputs, upscale_factor, \
               num_filters=64, num_res_blocks=16, \
               output_activation='tanh', act_fn='prelu', \
               relu_leakiness=DEFAULT_RELU_LEAKINESS, \
               use_norm_layers='not-first', norm_layer='batch'):
    
    super(Generator, self).__init__()
    upscale_factor = int(upscale_factor)
    assert (upscale_factor % 4 == 0)
    in_channels = num_inputs

    initial_conv = [nn.ZeroPad2d(4),
                    nn.Conv2d(in_channels, num_filters, kernel_size=9,
                              stride=1), 
                    nn.PReLU(num_parameters=64, init=0.1)]
    in_channels = num_filters

    if use_norm_layers != 'not-first' and use_norm_layers:
      initial_conv.append(nn.BatchNorm2d(64))
    elif use_norm_layers == 'not-first':
      use_norm_layers = True

    res_blocks = []
    for idx in range(num_res_blocks):
      res_blocks += [ResBlock(in_channels, num_filters, kernel_size=3,
                              use_norm_layers=use_norm_layers,
                              norm_layer=norm_layer, act_fn=act_fn,
                              relu_leakiness=relu_leakiness)]

    second_conv = [nn.ZeroPad2d(1),
                   nn.Conv2d(in_channels, num_filters, kernel_size=3,
                             stride=1)]
    in_channels = num_filters
    if use_norm_layers:
      second_conv.append(nn.BatchNorm2d(in_channels))

    upsample = []
    if upscale_factor > 1:
      scale = 2 if upscale_factor % 2 == 0 else 3
      for idx in range(upscale_factor // scale):
        upsample += [ nn.ZeroPad2d(1),
                     nn.Conv2d(in_channels, scale * scale * 256,
                               kernel_size=3, stride=1),
                     nn.PixelShuffle(upscale_factor=scale),
                     nn.PReLU(num_parameters=256, init=0.1)]
        in_channels = 256

    final_conv = [nn.ZeroPad2d(4),
                  nn.Conv2d(in_channels, num_outputs, kernel_size=9,
                            stride=1)]
    if output_activation != 'none':
      final_conv.append(nn.Tanh())

    self.initial_conv = nn.Sequential(*initial_conv)
    self.body = nn.Sequential(*(res_blocks + second_conv))
    self.upsample = nn.Sequential(*upsample)
    self.final_conv = nn.Sequential(*final_conv)
    self.output_activation = output_activation

  def forward(self, x):
    initial = self.initial_conv(x)
    x = self.body(initial)
    x = self.upsample(x + initial)
    x = self.final_conv(x)
    return x


class ResBlock(nn.Module):
  def __init__(self, in_channels, num_filters, kernel_size, use_norm_layers,
               norm_layer, act_fn, relu_leakiness, padding='zero'):
    
    super(ResBlock, self).__init__()
    modules = [nn.ZeroPad2d(1),
               nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                         stride=1)]
    if use_norm_layers:
      modules.append(nn.BatchNorm2d(64))

    modules += [nn.PReLU(num_parameters=64, init=0.1),
                nn.ZeroPad2d(1),
                nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size,
                          stride=1)]
    if use_norm_layers:
      modules.append(nn.BatchNorm2d(64))

    self.block = nn.Sequential(*modules)

  def forward(self, x):
    return self.block(x) + x


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
            nn.Linear(6 * 6 * 512, 1024),  # bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)  # bias=True)
            # nn.Sigmoid()
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
        b = b.view(b.shape[0], -1) 
        b = self.layer9(b)
        b = torch.sigmoid(b)  

        return b


class Dataset(Dataset): 
    def __init__(self, direct, crop, resolution):
        super(Dataset, self).__init__()
        self.dataset = [os.path.join(direct, x) for x in os.listdir(direct)]  
        self.lr_transform = self.LR_crop(crop, resolution)
        self.hr_transform = self.HR_crop(crop)

    def __getitem__(self, index):
        path = self.dataset[index]
        hr_img = self.hr_transform(Image.open(path).convert('RGB')) 
        lr_img = self.lr_transform(hr_img)  
        hr_img = hr_img * 2 - 1  

        return lr_img, hr_img

    def __len__(self):
        return len(self.dataset)

    def LR_crop(self, crop, interpolation_scale):
        data_transform_lr = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize(
                                                    (crop // interpolation_scale, crop // interpolation_scale), \
                                                    interpolation=Image.BICUBIC),
                                                transforms.ToTensor()])
        return data_transform_lr

    def HR_crop(self, crop):
        data_transform_hr = transforms.Compose([transforms.RandomCrop(crop), 
                                                transforms.ToTensor()])
        return data_transform_hr
    
class Dataset_Test(Dataset):
    def __init__(self, direct, crop):
        super(Dataset_Test, self).__init__()
        
        self.image_filenames = [os.path.join(direct, x) for x in os.listdir(direct)]
        self.upscale_factor = crop

    def __getitem__(self, index):
        img_name = self.image_filenames[index]
        img = Image.open(img_name).convert('RGB')
        hr_image = transforms.ToTensor()(img)
        if hr_image.size(1) % self.upscale_factor != 0 and hr_image.size(2) % self.upscale_factor != 0:
            height_diff = hr_image.size(1) % self.upscale_factor
            width_diff = hr_image.size(2) % self.upscale_factor
            hr_image = self.HR_crop(hr_image.size(1) - height_diff, hr_image.size(2) - width_diff)(hr_image)

        elif hr_image.size(1) % self.upscale_factor != 0 and hr_image.size(2) % self.upscale_factor == 0:
            height_diff = hr_image.size(1) % self.upscale_factor
            hr_image = self.HR_crop(hr_image.size(1) - height_diff, hr_image.size(2))(hr_image)

        elif hr_image.size(1) % self.upscale_factor == 0 and hr_image.size(2) % self.upscale_factor != 0:
            width_diff = hr_image.size(2) % self.upscale_factor
            hr_image = self.HR_crop(hr_image.size(1), hr_image.size(2) - width_diff)(hr_image)

        lr_transform = self.LR_crop(
            h=hr_image.size(1),
            w=hr_image.size(2),
            interpolation_scale=self.upscale_factor
        )
        lr_image = lr_transform(hr_image)
        hr_image = (hr_image * 2) - 1
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

    def LR_crop(self, h, w, interpolation_scale):
        data_transform_lr = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((h//interpolation_scale, w//interpolation_scale), \
                                                                  interpolation=Image.BICUBIC),
                                                transforms.ToTensor()]) 
        return data_transform_lr

    def HR_crop(self, h, w):
        data_transform_hr = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((h, w), interpolation=Image.BICUBIC),
                                                transforms.ToTensor()])
        return data_transform_hr


class Loss_Generator(nn.Module):
    def __init__(self):
        super(Loss_Generator, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.mse_loss = nn.MSELoss()
        self.bceloss = nn.BCELoss()

        blocks = []
        blocks.append(vgg19.features[:4].eval())
        blocks.append(vgg19.features[4:9].eval())
        blocks.append(vgg19.features[9:18].eval())
        blocks.append(vgg19.features[18:27].eval())
        blocks.append(vgg19.features[27:36].eval())

        blocks = nn.ModuleList(blocks)

        for bl in blocks.parameters():
            bl.requires_grad = False
        self.blocks = blocks
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) 
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, hr_, sr_, out_discriminator):
        content_loss = 0.0
        output_x, output_y = [], []
        x = torch.clone(hr_)
        y = torch.clone(sr_)
        x = x.sub(self.mean).div(self.std)
        y = y.sub(self.mean).div(self.std)
        for block in self.blocks:
            x = block(x).div(12.75)
            y = block(y).div(12.75)
            content_loss += self.mse_loss(x, y) 
        ones = torch.ones_like(out_discriminator)
        ones = ones
        adversarial_loss = self.bceloss(out_discriminator, ones)  
        beta = 0.001

        return content_loss + beta * adversarial_loss, adversarial_loss, content_loss


class Loss_Discriminator(nn.Module):
    def __init__(self):
        super(Loss_Discriminator, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, real, fake):
        one = torch.ones_like(real)
        one = one
        zeros = torch.zeros_like(fake)

        real_loss = self.loss(real, one)  
        fake_loss = self.loss(fake, zeros)  

        return fake_loss + real_loss


use_gpu = torch.cuda.is_available() 
device = torch.device('cuda:0' if use_gpu else 'cpu')
print('use_gpu: ', use_gpu)

batch_size = 16
learning_rateD = 0.0001 
learning_rateG = 0.0004  
num_epoch = 20000

root_t = '/content/drive/MyDrive/T'  
crop_size = 96
scale = 4

ep = []

train_dataset = Dataset(root_t, crop_size, scale)
train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = Dataset_Test(root_v, scale)
val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

print('LENGHT TRAIN DATASET: ', len(train_dataset))
print('LENGHT VAL DATASET: ', len(val_dataset))

dataloader = {'train': train, 'val':val} 

net_G = Generator()
net_D = Discriminator()

if use_gpu: 
    net_G.to(device)
    net_D.to(device)

state_dict = torch.load('/content/drive/MyDrive/data_download/dataset_images/model_best.pth')
net_D.load_state_dict(state_dict['D'])
net_G.load_state_dict(state_dict['G'])

cur_e = state_dict['epoch']

criterion_G = Loss_Generator().to(device) if use_gpu else Loss_Generator()
criterion_D = Loss_Discriminator().to(device) if use_gpu else Loss_Discriminator()
optimizer_G = optim.Adam(net_G.parameters(), lr=learning_rateG, betas=(0.9, 0.999))  
optimizer_D = optim.Adam(net_D.parameters(), lr=learning_rateD, betas=(0.9, 0.999))

scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=0.1, patience=1000, verbose=True)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, factor=0.1, patience=1000, verbose=True)

optimizer_G.load_state_dict(state_dict['optimizerG_state_dict'])
optimizer_D.load_state_dict(state_dict['optimizerD_state_dict'])


% cd / content / drive / MyDrive
% load_ext
tensorboard
current_time = str(datetime.datetime.now().timestamp())
train_log_dir = 'logs/tensorboard/train/color'
train_summary_writer = summary.create_file_writer(train_log_dir)

% tensorboard - -logdir logs / tensorboard

picture_t, picture_v = 0, 0
k_t, k_v = 0, 0

los_sum = 0.0
loss_g = 0.0
step_d = 0
best_psnr = -100



for epoch in tqdm(range(cur_e, num_epoch)):
    l_G, l_D = 0, 0
    for phase in ['train', 'val']:
        if phase == 'train':
            net_G.train()
            net_D.train()
        else:
            net_G.eval()
            net_D.eval()

        z = 0 

        for i, (lr_img, hr_img) in enumerate(dataloader[phase]):

            lr_img = lr_img.clone().detach().to(device) if use_gpu else lr_img.clone().detach()
            hr_img = hr_img.clone().detach().to(device) if use_gpu else hr_img.clone().detach()

            with torch.set_grad_enabled(phase == 'train'):

                # TRAIN DISCRIMINATOR
                try:
                    if loss_d.item() >= 0.4 and phase == 'train':
                        net_D.zero_grad()
                except:
                    {}

                sr_img = net_G(lr_img)

                real_out = net_D(hr_img)
                fake_out = net_D(sr_img)
                loss_d = criterion_D(real_out, fake_out)

                if loss_d.item() >= 0.4 and phase == 'train':
                    loss_d.backward(retain_graph=True)
                    optimizer_D.step()
                    los_sum += float(loss_d)
                    step_d += 1

                # TRAIN GENERATOR
                if phase == 'train':
                    net_G.zero_grad()
                fake_out = net_D(sr_img) 
                loss_g, adversarial_loss, content_loss = criterion_G(hr_img, sr_img, fake_out)

                if phase == 'train':
                    loss_g.backward() 
                    optimizer_G.step()

                # counting metrics and output to tensorboard
                if phase == 'train':
                    SR = (sr_img[0] + 1) / 2
                    SR = SR.cpu().detach().numpy()
                    SR = SR.transpose(1, 2, 0)
                    HR = (hr_img[0] + 1) / 2
                    HR = HR.cpu().detach().numpy()
                    HR = HR.transpose(1, 2, 0)
                    PSNR = compare_psnr(HR, SR, data_range=255)
                    PIXEL_MAX = 255
                    (score, diff) = compare_ssim(SR, HR, full=True, multichannel=True)
                    
                    with train_summary_writer.as_default():
                        summary.scalar('train_loss_d', loss_d.item(), step=k_t)
                        summary.scalar('train_loss_g', loss_g.item(), step=k_t)
                        summary.scalar('train_advers_loss', adversarial_loss.item(), step=k_t)
                        summary.scalar('train_content_loss', content_loss.item(), step=k_t)
                        summary.scalar('train_psnr', PSNR, step=k_t)
                        summary.scalar('train_ssim', score, step=k_t)

                    k_t += 1
                    z += 1

                if i % 50 == 0:
                    print('\nPhase[{}], Epoch [{}/{}], Step [{}], Loss_D: {:.5f}, Loss_G: {:.5f}'.format(phase,
                                                                                                         epoch + 1,
                                                                                                         num_epoch,
                                                                                                         i + 1,
                                                                                                         loss_d.item(),
                                                                                                         loss_g.item()))
                    print("GRADIENT NORM METHOD_G: {0}".format(torch.norm(net_G.layer21.weight.grad)))
                    print("GRADIENT NORM METHOD_D: {0}".format(torch.norm(net_D.layer9[2].weight.grad)))

                l_G = l_G + loss_g.item()
                l_D = l_D + loss_d.item()

                if (i + 1) % 1000 == 0 and phase == 'train':
                    state = {
                        'epoch': epoch,
                        'step': k_t,
                        'D': net_D.state_dict(),
                        'G': net_G.state_dict(),
                        'optimizerD_state_dict': optimizer_D.state_dict(),
                        'optimizerG_state_dict': optimizer_G.state_dict()}
                    torch.save(state, '/content/drive/MyDrive/data_download/dataset_images/model.pth')

                    for j in range(4):
                        save_path = '/content/drive/MyDrive/data_out_train/'
                        save_name = 'img-{}.png'.format(picture_t)
                        save_hr = 'img-{}.png'.format(picture_t + 1)
                        picture_t += 2
                        save_image(sr_img[j], f'{save_path}{save_name}', nrow=0, padding=0,
                                   normalize=True)
                        save_image(hr_img[j], f'{save_path}{save_hr}', nrow=0, padding=0, normalize=True, range=(-1, 1))
                
                if phase == 'val':
                    for j in range(6):
                        save_path = '/content/drive/MyDrive/data_out_val/'
                        save_sr = 'img-{}.png'.format(picture_v)
                        save_hr = 'img-{}.png'.format(picture_v + 1)
                        save_lr = 'img-{}.png'.format(picture_v + 2)
                        picture_v += 3
                        save_image(sr_img[j], f'{save_path}{save_sr}', nrow=0, padding=0, normalize=True) 
                        save_image(hr_img[j], f'{save_path}{save_hr}', nrow=0, padding=0, normalize=True)
                        save_image(lr_img[j], f'{save_path}{save_lr}', nrow=0, padding=0, normalize=True)

                if best_psnr < PSNR and phase == 'train':
                    state = {
                        'epoch': epoch,
                        'step': k_t,
                        'D': net_D.state_dict(),
                        'G': net_G.state_dict(),
                        'optimizerD_state_dict': optimizer_D.state_dict(),
                        'optimizerG_state_dict': optimizer_G.state_dict()
                    }
                    torch.save(state, '/content/drive/MyDrive/data_download/dataset_images/model_best.pth')
                
                if phase == 'train':
                    scheduler_G.step(l_G / (i + 1))
                    scheduler_D.step(l_D / (i + 1))
         
