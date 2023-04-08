# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import re
import itertools
from torch.autograd import Variable
from tqdm import *
import random
from models import  DCGAN_G, DCGAN_D,weights_init

parser = argparse.ArgumentParser(description='CIFAR-10 Training')
parser.add_argument('--lr', default=2e-4, type=float, help='learning_rate')
parser.add_argument('--model_dir', default='./tmp/', type=str, help='trained model saving path')
parser.add_argument('--image_dir', default='./tmp/', type=str, help='image saving path')
parser.add_argument('--data_path', default='./tmp/', type=str, help='image saving path')
parser.add_argument('--code_length', default= 128, type=int, help='noise code length')
parser.add_argument('--outcomes',  default= 8 , type=int, help='output number')
parser.add_argument('--G_split_num',  default= 5 , type=int, help='Generator training once per Discriminator training G_split_num times ')
parser.add_argument('--epoch', default=2000, type=int, help='training epoch')
parser.add_argument('--batch_size', default=256, type=int, help='training batch_size')
parser.add_argument('--beta1',default=0.5,type=float, help='beta1 for adam')
parser.add_argument('--beta2',default=0.9,type=float, help='beta2 for adam')
parser.add_argument('--entropy_lr',default=1.0,type=float, help='learning rate for entropy')
parser.add_argument('--buffer_size',default=1024,type=int,help='the size of buffer size')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Dataset

image_size=32

transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=False, transform=transform_train)

ConData = torch.utils.data.ConcatDataset([trainset,testset])

trainloader = torch.utils.data.DataLoader(ConData, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True,drop_last=True)
testloader = torch.utils.data.DataLoader(ConData, batch_size=1, shuffle=True, num_workers=0)

# Model Initialization

ger = DCGAN_G(input_z=args.code_length)
ger.apply(weights_init)
cls = DCGAN_D(num_outcomes=args.outcomes)
cls.apply(weights_init)


if use_cuda:
    ger.cuda()
    cls.cuda()

parameters_D = cls.parameters()
parameters_G = ger.parameters()

optimizer_G = torch.optim.Adam(parameters_G, lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(parameters_D, lr=args.lr, betas=(args.beta1, args.beta2))

g_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, 50, gamma=0.9, last_epoch=-1 )
d_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_D, 50, gamma=0.9, last_epoch=-1 )

def data_aug(tensor):
    dim = random.choice([2,3])
    tensor = torch.flip(tensor,[dim])
    tensor = tensor + 0.001 * torch.randn_like(tensor)
    return tensor

def entropy_buffer(buffer_size = args.buffer_size):
    replay_buffer = torch.FloatTensor(buffer_size, args.outcomes).uniform_(-1, 1)
    replay_buffer = torch.stack([replay_buffer.detach() for i in range(args.batch_size)], dim = 1)
    return replay_buffer


replay_buffer_generate = entropy_buffer().cuda()
replay_buffer_real = entropy_buffer().cuda()

def enrtopy_loss(batch_code,replay_buffer):

    batch_code_r = batch_code.unsqueeze(0)
    dist_metric = torch.mean(torch.pow((replay_buffer-batch_code_r),2),dim=2)

    dist_min,idx = torch.min(dist_metric,dim=0)

    loss = torch.mean(dist_min)

    batch_code_s = torch.stack([batch_code.detach()],dim=0)

    replay_buffer = torch.cat((replay_buffer[args.batch_size:],batch_code_s),dim=0)

    return loss,replay_buffer


def training_ours(epoch):
    global replay_buffer_generate
    global replay_buffer_real
    cls.train()
    ger.train()

    t = tqdm(trainloader)
    t.set_description("Epoch [{}/{}]".format(epoch,args.epoch))
    for batch_idx, (realdata,_) in enumerate(t):

        if use_cuda:
            realdata = realdata.cuda()

        input = torch.randn(realdata.size(0), args.code_length).cuda()

        optimizer_D.zero_grad()

        fake_imgs = ger(input).detach()


        t_z = cls(realdata)

        f_z = cls(fake_imgs)


        loss_regular =  torch.tensor([0])

# ==========================DLLE===============================
        realdata_aug = data_aug(realdata)
        fake_imgs_aug = data_aug(fake_imgs)
        t_z_aug = cls(realdata_aug)
        f_z_aug = cls(fake_imgs_aug)
        loss_dlle = F.mse_loss(t_z,t_z_aug.detach()) + F.mse_loss(f_z,f_z_aug.detach())


# ==========================DLLE===============================
        if epoch < 100:
            lr = epoch/100. * args.entropy_lr
        else:
            lr =  1.0 * args.entropy_lr
        loss_entropy, replay_buffer_real = enrtopy_loss(t_z,replay_buffer_real)
        loss_entropy =  -lr * torch.log(loss_entropy)

# ==========================L_MaF===============================
        loss_D = -torch.mean(t_z) + torch.mean(f_z)

# ==========================L_DIsoMap===============================

        alpha = torch.rand(realdata.size(0),1).cuda()

        mix_z = alpha * t_z + f_z * (1-alpha)

        mix_z = mix_z + torch.normal(mean = 0., std = 0.05, size= (mix_z.size())).cuda()

        alpha = alpha.view(-1,1,1,1)

        mix_input = alpha * realdata + (1-alpha) * fake_imgs

        loss_iso = F.mse_loss(cls(mix_input),mix_z.detach())

# ==========================Update===============================

        loss_dlle.backward(retain_graph=True)

        loss_entropy.backward(retain_graph=True)

        loss_iso.backward(retain_graph=True)

        loss_D.backward()

        optimizer_D.step()

        if batch_idx % args.G_split_num == 0:
            optimizer_G.zero_grad()
            g_imgs = ger(input)

            imgs_code = cls(g_imgs)

            loss_G = -torch.mean(imgs_code)
            loss_entropy= torch.tensor([0])
            if epoch < 100:
                lr = epoch/100. * args.entropy_lr
            else:
                lr =  1.0 * args.entropy_lr

            loss_entropy, replay_buffer_generate = enrtopy_loss(imgs_code,replay_buffer_generate)
            loss_entropy = -lr * torch.log(loss_entropy)

            loss_entropy.backward(retain_graph=True)

            loss_G.backward()
            optimizer_G.step()

            t.set_postfix_str('loss_G: {:4f}, loss_iso: {:4f} loss_D: {:4f} loss_DLLE:{:4f} loss_entropy:{:4f}'.format(
            loss_G.item(), loss_iso.item(),loss_D.item(),loss_dlle.item(), loss_entropy.item()
            ))
            t.update()

unloader = transforms.ToPILImage()

def save_images(image,name):
    image = image.cpu().clone()
    image = image * 0.5 + 0.5
    image = torch.clamp(image, min=0, max=1)
    image = unloader(image)
    image.save(name)

def save_model(dir_name,name):
    torch.save(ger.state_dict(),os.path.join(dir_name,name)+'gen.pth')
    torch.save(cls.state_dict(),os.path.join(dir_name,name)+'cls.pth')

def make_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)


def test(epoch,sample_num=1000):
    ger.eval()
    testloader = torch.utils.data.DataLoader(ConData, batch_size=1, shuffle=True, num_workers=1)
    with torch.no_grad():
        for batch_idx, (realdata, _) in enumerate(testloader):
            if batch_idx >=sample_num:
                break
            z = torch.randn(realdata.shape[0], args.code_length).cuda()

            outputs = ger(z)
            outputs = outputs.squeeze(0)

            test_image_dir = os.path.join(args.image_dir, str(epoch)+'/test/')
            make_dir(test_image_dir)
            name = os.path.join(test_image_dir,str(batch_idx)+'.png')
            save_images(outputs,name)


            realdata = realdata.squeeze(0)
            gt_image_dir = os.path.join(args.image_dir, str(epoch)+'/label/')
            make_dir(gt_image_dir)
            name = os.path.join(gt_image_dir,str(batch_idx)+'.png')
            save_images(realdata,name)

    make_dir(args.model_dir)
    save_model(args.model_dir,str(epoch))

if __name__ == '__main__':
    save_epoch = 50
    for e in range(args.epoch):
        training_ours(e)
        g_scheduler.step()
        d_scheduler.step()
        if e % save_epoch == 0:
            test(e)
