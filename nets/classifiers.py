# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from utils.utils import weights_init
import torch.nn.functional as F
from constants import *

class _netD_mnist(nn.Module):
    def __init__(self):
        super(_netD_mnist, self).__init__()
        nc = nc_netD
        ndf = ndf_netD
        self.conv2 = nn.Conv2d(nc, ndf, 4, 2, 1) #ndf*14*14
        #self.conv2 = nn.Conv2d(ndf, ndf, 1, 1, 0) #ndf *14*14
        self.pool = nn.MaxPool2d(2)# ndf * 7 * 7
        self.conv3 = nn.Conv2d(ndf, ndf*2, 3, 2, 1) #ndf*2 *4 * 4
        self.pool2 = nn.MaxPool2d(2) #ndf*2 *2*2
        self.fc_drop = nn.Dropout(p=0.5)
        self.linear = nn.Linear(ndf*8,10)
        self.apply(weights_init)

    def forward(self, input):
        #x = self.conv1(input)
        x = F.relu(self.pool(self.conv2(input)))
        x = F.relu(self.pool2(self.conv3(x)))
        x = x.view(x.size()[0], -1)
        x = self.linear(self.fc_drop(x))
        return x


class _netG_mnist(nn.Module):
    def __init__(self):
        super(_netG_mnist, self).__init__()
        nc = nc_netG
        ndf = ndf_netG
        self.conv1 = nn.Conv2d(nc, ndf, 5, 1, 2)
        self.conv2 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.conv3 = nn.Conv2d(ndf, ndf, 1, 1, 0)
        self.conv4 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.conv5 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        #self.conv6 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.conv7 = nn.Conv2d(ndf, ndf, 1, 1, 0)
        self.conv8 = nn.Conv2d(ndf, nc, 1, 1, 0)

        self.apply(weights_init)
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.tanh(x)
        return x




class _netG_cifar10(nn.Module):
    def __init__(self):
        super(_netG_cifar10, self).__init__()
        nc = nc_netG
        ndf = ndf_netG
        self.conv1 = nn.Conv2d(nc, ndf, 5, 1, 2)
        self.conv2 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.conv3 = nn.Conv2d(ndf, ndf, 1, 1, 0)
        self.conv4 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.conv5 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        #self.conv6 = nn.Conv2d(ndf, ndf, 3, 1, 1)
        self.conv7 = nn.Conv2d(ndf, ndf, 1, 1, 0)
        self.conv8 = nn.Conv2d(ndf, nc, 1, 1, 0)

        self.apply(weights_init)
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = F.tanh(x)
        return x

'''
class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        nc = nc_netG
        ndf = ndf_netG
        ninput = ninput_netG
        ngf = ngf_netG
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 2)
        
        self.conv2 = nn.Conv2d(ndf, ndf/2, 4, 2, 2)
        self.pool2 = nn.MaxPool2d(2,2)
        # (ndf/2) * 4 * 4

        #self.norm = nn.InstanceNorm1d(1)
        self.conv = nn.Conv1d(1, 10*ngf, ninput,1,0)

        self.linear3 = nn.Linear(ngf, image_size)
        #self.linear4 = nn.Linear(ngf, image_size)
        self.apply(weights_init)
    def forward(self, input1, ind, input2):
        x = F.relu(self.conv1(input1), True)
        # ndf * 14 * 14
        x = self.pool2(F.relu(self.conv2(x), True))
        # (ndf/2) * 4 * 4
        #x = x.view(x.size()[0], 1, -1)
        #x = self.norm(x)
        x = x.view(x.size()[0],-1)
        x = torch.cat([x, input2], 1)
        x = x.view(x.size()[0], 1, -1)
        x = F.relu(self.conv(x))
        #10*ngf
        x = x.view(x.size()[0],10,-1)
        x = x.gather(1, ind)
        #1*ngf
        x = x.view(x.size()[0], -1)
        #ngf
        x = F.tanh(self.linear3(F.relu(x,True)))
        return x
'''





