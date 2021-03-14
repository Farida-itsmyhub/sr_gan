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

# from livelossplot import PlotLosses

# МАУНТ ДИСКА
# from google.colab import drive
# drive.mount('/content/drive')

# drive.mount('/content/gdrive')#, force_remount = True)
'''%cd /content/gdrive/MyDrive/data_download/
!pwd'''

# !wget http://data.csail.mit.edu/places/places365/test_256.tar
# !tar -xvf test_256.tar -C /My Drive/data_download/ #/dataset_images/
# !unzip -q -n "gdrive/My Drive/data_download/test_256.tar"
# !ls
# %cd ..
'''!cp test_256.tar /content
%cd ..
%cd ..
%cd ..
!tar -xvf test_256.tar '''


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()  # вызов init у родителя Generator = Module
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU())

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
            nn.BatchNorm2d(64, affine=True)
        )
        self.layer19 = Repiet_block(64)
        self.layer20 = Repiet_block(64)
        self.layer21 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        self.layer22 = nn.Tanh()  # Sigmoid()   # новый слой (нет в статье) - тк нужны значения [0;1]

        # mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1) ПОКА ЗАКОМЕНИТИЛА - НЕПОНЯТНО ПОРИСХОЖДЕНИЕ
        # std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        # self.register_buffer('mean', mean)
        # self.register_buffer('std', std)

    def forward(self, a):
        # a = (a - self.mean) / self.std
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
        b = self.layer22(b)  # ПОКА ЧТО УБРАЛА

        return b


class Residual_block(nn.Module):
    def __init__(self, numb):
        super(Residual_block, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=numb, out_channels=numb, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.BatchNorm2d(numb, affine=True)
        self.layer3 = nn.PReLU()
        self.layer4 = nn.Conv2d(in_channels=numb, out_channels=numb, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.BatchNorm2d(numb, affine=True)

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
        self.layer3 = nn.PReLU()  # init=0.2

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
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
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
        b = b.view(b.shape[0], -1)  # b.reshape(b.size(0), -1)
        # b = b.view(-1, 1).squeeze(1)
        b = self.layer9(b)

        # ba = b.size(0) #  b.shape[0]
        b = torch.sigmoid(b)  # .view(ba))  # не было '-1'

        return b


class Dataset(Dataset):  # Dataset
    def __init__(self, direct, crop, resolution):
        super(Dataset, self).__init__()
        self.dataset = [os.path.join(direct, x) for x in os.listdir(direct)]  # проверку на формат jpg, png ... ?
        self.lr_transform = self.LR_crop(crop, resolution)
        self.hr_transform = self.HR_crop(crop)

    def __getitem__(self, index):
        path = self.dataset[index]
        hr_img = self.hr_transform(Image.open(path).convert('RGB'))  # "L"))
        lr_img = self.lr_transform(hr_img)  # Image.open(path).convert("RGB"))  # [0; 1]

        # для просмотра в plot
        '''image = lr_img.cpu().detach().numpy()
        image1 = image.transpose(1, 2, 0)
        plt.imshow(image1)
        plt.show()
        image = hr_img.cpu().detach().numpy()
        image2 = image.transpose(1, 2, 0)
        plt.imshow(image2)
        plt.show()'''

        hr_img = hr_img * 2 - 1  # [-1; 1]

        # image = ((hr_img+1)/2).cpu().detach().numpy()
        # image = image.transpose(1, 2, 0)
        # plt.imsave(arr=image, fname='{}{}'.format('gdrive/My Drive/data_out/', 'yuyuy.jpg'))

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
        data_transform_hr = transforms.Compose([transforms.RandomCrop(crop),  # не нужно преобразовывать в PIL
                                                # transforms.Resize(crop), #, interpolation=Image.BICUBIC),
                                                transforms.ToTensor()])
        return data_transform_hr


class Loss_Generator(nn.Module):
    def __init__(self):
        super(Loss_Generator, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        # self.loss_vgg19 = nn.Sequential(*list(vgg19.children()))[0]  # children : все слои по порядку
        self.mse_loss = nn.MSELoss()
        # self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss()

        blocks = []
        # blocks.append(vgg19.features[:36].eval())
        blocks.append(vgg19.features[:4].eval())
        blocks.append(vgg19.features[4:9].eval())
        blocks.append(vgg19.features[9:18].eval())
        blocks.append(vgg19.features[18:27].eval())
        blocks.append(vgg19.features[27:36].eval())

        blocks = nn.ModuleList(blocks)

        for bl in blocks.parameters():
            bl.requires_grad = False
        self.blocks = blocks
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1,
                                                        1)  # torch.Tensor([-0.03, -0.088, -0.188]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # torch.Tensor([0.458, 0.448, 0.45]).view(1,3,1,1)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, hr_, sr_, out_discriminator):
        # content_loss = (self.mse_loss(self.loss_vgg19(hr_img)/12.75, self.loss_vgg19(sr_img)/12.75))
        content_loss = 0.0
        output_x, output_y = [], []

        # x = (hr_ + 1) / 2  #ЗАКОМЕНТИЛА
        # y = (sr_ + 1) / 2  # попробовать ниче не делать
        # source_range=(-1., 1.)
        # x = x.clamp(source_range[0], source_range[1]) #(sr_ + 1)/2  hr_
        # y = y.clamp(source_range[0], source_range[1]) #(sr_ + 1)/2
        x = torch.clone(hr_)
        y = torch.clone(sr_)
        x = x.sub(self.mean).div(self.std)
        y = y.sub(self.mean).div(self.std)
        # x = (hr_img- self.mean) / self.std  # ПОКА ЗАКОМЕНТИЛА
        # y = (sr_img - self.mean) / self.std
        for block in self.blocks:
            x = block(x).div(12.75)
            y = block(y).div(12.75)
            '''x = block(hr_img).div(12.75)    проверить обновляются ли веса после обновления!! 
            y = block(sr_img).div(12.75)'''
            content_loss += self.mse_loss(x, y)  # попробовать по другому (штрафы) !!

        # content_loss = content_loss.div(len(self.blocks))
        # logit = self.sigmoid(out_discriminator)
        ones = torch.ones_like(out_discriminator)
        ones = ones
        adversarial_loss = self.bceloss(out_discriminator,
                                        ones)  # torch.ones_like(out_discriminator)) ДОБАВИТЬ ИЛИ УБРАТЬ: torch.sigmoid
        beta = 0.001

        return content_loss + beta * adversarial_loss, adversarial_loss, content_loss


class Loss_Discriminator(nn.Module):
    def __init__(self):
        super(Loss_Discriminator, self).__init__()
        # self.sigmoid = nn.Sigmoid()  #nn.CrossEntropyLoss().to(device) if use_gpu else nn.CrossEntropyLoss()  #nn.LogSigmoid()
        self.loss = nn.BCELoss()

    def forward(self, real, fake):
        # logits_f = self.sigmoid(fake)  # я убрала логиты (типо зажимы)
        # logits_r = self.sigmoid(real)
        one = torch.ones_like(real)
        one = one
        zeros = torch.zeros_like(fake)
        # zeros = zeros + 0.1

        real_loss = self.loss(real,
                              one)  # torch.ones_like(real))# torch.sigmoid(logits_r), torch.ones_like(real))#, torch.ones_like(real))  #self.logSigmoid(real)
        fake_loss = self.loss(fake, zeros)  # torch.zeros_like(fake))#torch.sigmoid(logits_f), torch.zeros_like(fake))

        return fake_loss + real_loss


def log_gradients(gradmap, step):
    for k, v in gradmap.items():
        experiment.log_histogram_3d(to_numpy(v), name=k, step=step)


def log_weights(model, step):
    for name, layer in zip(model._modules, model.children()):
        if "activ" in name:
            continue

        if not hasattr(layer, "weight"):
            continue

        wname = "%s.%s" % (name, "weight")
        bname = "%s.%s" % (name, "bias")

        experiment.log_histogram_3d(to_numpy(layer.weight), name=wname, step=step)
        experiment.log_histogram_3d(to_numpy(layer.bias), name=bname, step=step)


use_gpu = torch.cuda.is_available()  # использование gpu
device = torch.device('cuda:0' if use_gpu else 'cpu')
print('use_gpu: ', use_gpu)

batch_size = 16
learning_rateD = 0.0001  # 0.000001
learning_rateG = 0.0004  # 0.00001
# learning_rateD = 0.0001 #0.000001  НА 0.001 !!
# learning_rateG = 0.001 #0.00001
num_epoch = 20000

root_t = '/content/drive/MyDrive/T'  # data_download/dataset_images/test_256' #data_download/dataset_images/test_256'  # div2k'  #'/content/drive/My Drive/data'  #'/content/test_256'  #'gdrive/My Drive/data' #'./data/'  #C:/Users/MM/PycharmProjects/SRgan - в начало
# root_v = '/content/drive/MyDrive/data_download/test_256_val'
crop_size = 96
scale = 4

ep = []

# print(os.listdir(root))
# print(torch.__version__, sys.version)

train_dataset = Dataset(root_t, crop_size, scale)
print(len(train_dataset))
train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = Dataset(root_v, crop_size, scale)
# val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

dataloader = {'train': train}  # , 'val':val}

net_G = Generator()
net_D = Discriminator()

if use_gpu:  # использоване gpu
    net_G.to(device)
    net_D.to(device)

state_dict = torch.load('/content/drive/MyDrive/data_download/dataset_images/model_apa.pth')
net_D.load_state_dict(state_dict['D'])
net_G.load_state_dict(state_dict['G'])  # , map_location=torch.device('cpu'))
cur_e = state_dict['epoch']

criterion_G = Loss_Generator().to(device) if use_gpu else Loss_Generator()
criterion_D = Loss_Discriminator().to(device) if use_gpu else Loss_Discriminator()
optimizer_G = optim.Adam(net_G.parameters(), lr=learning_rateG, betas=(0.9, 0.999))  # как работает ?
optimizer_D = optim.Adam(net_D.parameters(), lr=learning_rateD, betas=(0.9, 0.999))

scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, factor=0.1, patience=1000, verbose=True)
scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, factor=0.1, patience=1000, verbose=True)

optimizer_G.load_state_dict(state_dict['optimizerG_state_dict'])
optimizer_D.load_state_dict(state_dict['optimizerD_state_dict'])

for param_group in optimizer_D.param_groups:
    param_group['lr'] = 0.000001
for param_group in optimizer_G.param_groups:
    param_group['lr'] = 0.000004

# out_loss = open("/content/drive/MyDrive/data_download/history_loss_color.txt", "a")

# ТОЛЬКО ЭТО НАДО РАСКОММЕНТИТЬ (НИЖНЕЕ НЕ НАДО)
% cd / content / drive / MyDrive
% load_ext
tensorboard
current_time = str(datetime.datetime.now().timestamp())
train_log_dir = 'logs/tensorboard/train/color'  # test256' #+ current_time  #1612986651.892884'  black_white
# test_log_dir = 'storage/test/' + current_time
train_summary_writer = summary.create_file_writer(train_log_dir)
# test_summary_writer = summary.create_file_writer(test_log_dir)

% tensorboard - -logdir
logs / tensorboard

picture_t, picture_v = 0, 0
k_t, k_v = 0, 0

los_sum = 0.0
loss_g = 0.0
step_d = 0
best_psnr = -100

# !pip install livelossplot --quiet
# logs = {}
# liveloss = PlotLosses()

! pip
install - -upgrade
comet_ml
from comet_ml import Experiment

experiment = Experiment(api_key="<YOUR_API_KEY>", project_name="histograms")

for epoch in tqdm(range(cur_e, num_epoch)):
    gradmap = {}
    l_G, l_D = 0, 0
    for phase in ['train']:  # , 'val']:
        if phase == 'train':
            net_G.train()
            net_D.train()
        else:
            net_G.eval()
            net_D.eval()

        z = 0  # epoch_psnr

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
                fake_out = net_D(sr_img)  # МОЖЕТ УБРАТЬ, А ???
                loss_g, adversarial_loss, content_loss = criterion_G(hr_img, sr_img, fake_out)

                if phase == 'train':
                    loss_g.backward()  # retain_graph=True)
                    optimizer_G.step()

                if phase == 'train':
                    # ep.append(k)
                    # out_loss.write('{0} {1}\n'.format(loss_d.item(), loss_g.item()))
                    # min_v = torch.min(sr_img[0])
                    # range_v = torch.max(sr_img[0]) - min_v
                    # SR = (sr_img[0] - min_v) / range_v
                    SR = (sr_img[0] + 1) / 2
                    SR = SR.cpu().detach().numpy()
                    SR = SR.transpose(1, 2, 0)
                    HR = (hr_img[0] + 1) / 2
                    HR = HR.cpu().detach().numpy()
                    HR = HR.transpose(1, 2, 0)
                    PSNR = compare_psnr(HR, SR, data_range=255)
                    PIXEL_MAX = 255
                    (score, diff) = compare_ssim(SR, HR, full=True, multichannel=True)
                    # TENSORBOARD
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

                gradmap = update_gradient_map(net_D, gradmap)

                l_G = l_G + loss_g.item()
                l_D = l_D + loss_d.item()

                if (i + 1) % 400 == 0 and phase == 'train':
                    state = {
                        'epoch': epoch,
                        'step': k_t,
                        'D': net_D.state_dict(),
                        'G': net_G.state_dict(),
                        'optimizerD_state_dict': optimizer_D.state_dict(),
                        'optimizerG_state_dict': optimizer_G.state_dict()}
                    torch.save(state, '/content/drive/MyDrive/data_download/dataset_images/model_7.pth')

                if (i + 1) % 700 == 0 and phase == 'train':
                    state = {
                        'epoch': epoch,
                        'step': k_t,
                        'D': net_D.state_dict(),
                        'G': net_G.state_dict(),
                        'optimizerD_state_dict': optimizer_D.state_dict(),
                        'optimizerG_state_dict': optimizer_G.state_dict()
                    }
                    torch.save(state, '/content/drive/MyDrive/data_download/dataset_images/model_6.pth')

                    for j in range(4):
                        save_path = '/content/drive/MyDrive/data_out_train/'
                        save_name = 'img-{}.png'.format(picture_t)
                        save_hr = 'img-{}.png'.format(picture_t + 1)
                        picture_t += 2
                        save_image(sr_img[j], f'{save_path}{save_name}', nrow=0, padding=0,
                                   normalize=True)  # , range=(-1, 1))
                        save_image(hr_img[j], f'{save_path}{save_hr}', nrow=0, padding=0, normalize=True, range=(-1, 1))
                if phase == 'train':
                    scheduler_G.step(l_G / (i + 1))
                    scheduler_D.step(l_D / (i + 1))
                    # epoch_psnr += PSNR

        # scale gradients
        for k, v in gradmap.items():
            gradmap[k] = v  # / steps_per_epoch

        log_gradients(gradmap, epoch)
        log_weights(net_D, epoch)  # * steps_per_epoch)
