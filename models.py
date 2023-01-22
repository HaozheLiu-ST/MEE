import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import sys
import numpy as np
import copy
import math
from torch.autograd.function import InplaceFunction
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def mixup_process(out, indices, lam):
    return out*lam + out[indices]*(1-lam)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def Hloss(res):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    b = S(res) * LS(res)
    b = -1 * torch.mean(b)
    return b


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet_orthognal(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, head_ebm=16, dropout_rate=0, gain=1.0,):
        super(Wide_ResNet_orthognal, self).__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        #self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes, bias=False)

        self.ebm_linear = nn.Linear(nStages[3], head_ebm*2, bias=False)
        self.out_comes = head_ebm
        self.nChannels = nStages[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data,gain)   # Initializing with orthogonal rows

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _embedding(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0),-1)
        # out = F.normalize(out, dim=1, p=2)
        return out
    def classification(self,x):
        out =  self._embedding(x)
        return self.linear(output)
    def energy_score(self,x):
        out =  self._embedding(x)
        stat_tuple = self.ebm_linear(out).unsqueeze(2).unsqueeze(3)
        mu, logvar = stat_tuple.chunk(2, 1)
        std = logvar.mul(0.5).exp_()
        epsilon = torch.randn(x.shape[0], self.out_comes, 1, 1).to(stat_tuple)
        output = epsilon.mul(std).add_(mu).view(-1, self.out_comes)
        return output

class OverfittingNet(nn.Module):
    def __init__(self, net, mu, sigma):
        super(OverfittingNet, self).__init__()
        self.mu = torch.Tensor(mu).float().view(3, 1, 1).cuda()
        self.sigma = torch.Tensor(sigma).float().view(3, 1, 1).cuda()
        self.net = net
    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return self.net(x)
    def dynamic_aggregate(self,x1,x2,alpha,noise_std=0.05):
        x1 = (x1 - self.mu) / self.sigma
        x2 = (x2 - self.mu) / self.sigma
        return self.net.dynamic_aggregate(x1,x2,alpha=alpha,noise_std=noise_std)
    def embedding(self,x1):
        x1 = (x1 - self.mu) / self.sigma
        return self.net._embedding(x1)



import os
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as spectral_norm

class DCGAN_G(nn.Module):
    def __init__(self,image_size=32,input_z=128,G_h_size=128,n_channels=3):
        super(DCGAN_G, self).__init__()
        model = []
        self.input_z = input_z
        mult = image_size // 8
        # start block
        model.append(nn.ConvTranspose2d(input_z, G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
        model.append(nn.BatchNorm2d(G_h_size * mult))

        model.append(nn.ReLU())

        # middel block
        while mult > 1:
            model.append(nn.ConvTranspose2d(G_h_size * mult, G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
            model.append(nn.BatchNorm2d(G_h_size * (mult//2)))

            model.append(nn.ReLU())

            mult = mult // 2

        # end block
        model.append(nn.ConvTranspose2d(G_h_size, n_channels, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(nn.Tanh())

        self.model = nn.Sequential(*model)

    def forward(self, input):
        out = input.view(input.shape[0], self.input_z, 1, 1)
        output = self.model(out)
        return output


class DCGAN_D(nn.Module):
    def __init__(self, n_channels=3,D_h_size=128,image_size=32,num_outcomes=8,use_adaptive_reparam=True):
        super(DCGAN_D, self).__init__()
        model = []
        self.D_h_size = D_h_size
        self.num_outcomes = num_outcomes
        self.use_adaptive_reparam = use_adaptive_reparam
        # start block
        model.append(spectral_norm(nn.Conv2d(n_channels, D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        image_size_new = image_size // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(spectral_norm(nn.Conv2d(D_h_size * mult, D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.2, inplace=True))

            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(D_h_size * mult * 4 * 4)
        out_size = num_outcomes
        fc = nn.Linear(in_size, out_size, bias=False)
        nn.init.orthogonal_(fc.weight.data)
        self.fc = spectral_norm(fc)
        # resampling trick
        self.reparam = spectral_norm(nn.Linear(in_size, out_size * 2, bias=False))
        nn.init.orthogonal_(self.reparam.weight.data)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, input):
        y = self.model(input)

        y = y.view(-1, self.D_h_size * self.mult * 4 * 4)
        output = self.fc(y).view(-1, self.num_outcomes)

        # re-parameterization trick
        if self.use_adaptive_reparam:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(input.shape[0], self.num_outcomes, 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.num_outcomes)
        return output
    def embedding(self,input):
        y = self.model(input)
        y = y.view(-1, self.D_h_size * self.mult * 4 * 4)
        stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
        mu, logvar = stat_tuple.chunk(2, 1)

        std = logvar.mul(0.5).exp_()

        epsilon = torch.zeros(input.shape[0], self.num_outcomes, 1, 1).cuda()

        output = epsilon.add_(mu).view(-1, self.num_outcomes)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
