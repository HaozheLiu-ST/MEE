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
from torch.autograd import Variable
from tqdm import *

from models import DCGAN_G

parser = argparse.ArgumentParser(description='CIFAR-10 Training')

parser.add_argument('--model_path', default='./tmp/', type=str, help='trained model saving path')
parser.add_argument('--image_dir', default='./tmp/', type=str, help='image saving path')
parser.add_argument('--code_length', default= 64, type=int, help='noise code length')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()


image_size=32

transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)

ConData = torch.utils.data.ConcatDataset([trainset,testset])
testloader = torch.utils.data.DataLoader(ConData, batch_size=1, shuffle=True, num_workers=1)

ger = DCGAN_G(input_z=args.code_length)

ger.load_state_dict(torch.load(args.model_path))

if use_cuda:
    ger.cuda()

def test(sample_num=2000):
    ger.eval()
    with torch.no_grad():
        for batch_idx, (realdata, _) in enumerate(tqdm(testloader)):
            if batch_idx >= sample_num:
                break
            z = torch.randn(realdata.shape[0], args.code_length).cuda()

            outputs = ger(z)
            outputs = outputs.squeeze(0)

            test_image_dir = os.path.join(args.image_dir, 'test/')
            make_dir(test_image_dir)
            name = os.path.join(test_image_dir,str(batch_idx)+'.png')
            save_images(outputs,name)


            realdata = realdata.squeeze(0)
            gt_image_dir = os.path.join(args.image_dir,'label/')
            make_dir(gt_image_dir)
            name = os.path.join(gt_image_dir,str(batch_idx)+'.png')
            save_images(realdata,name)
def make_dir(name):
    if not os.path.isdir(name):
        os.makedirs(name)

unloader = transforms.ToPILImage()
def save_images(image,name):
    image = image.cpu().clone()
    image = image * 0.5 + 0.5
    image = torch.clamp(image, min=0, max=1)
    image = unloader(image)
    image.save(name)
if __name__ == '__main__':
    test(50000)
