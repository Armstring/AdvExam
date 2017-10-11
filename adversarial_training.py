from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim

from nets.classifiers import _netD_mnist,_netG_mnist
from constants import *
import models.model_train as model_train
from dataProcess.read_data import read_MNIST
import glob
from utils.utils import accu,TestAcc_dataloader, TestAcc_tensor


#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)

train_data , test_data = read_MNIST(batch_size, test_batch_size)

netD = _netD_mnist()
netD.cuda()
netD.load_state_dict(torch.load('netD.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc_dataloader(netD,test_data)))
########################

#model_train.adv_train_gradient(train_data, test_data, 'sign', coef_FGSM, 1)
#model_train.adv_train_gradient(train_data, test_data, 'sign', coef_FGSM, 3)
#model_train.adv_train_gradient(train_data, test_data, 2, coef_L2, 1)
#model_train.adv_train_gradient(train_data, test_data, 2, coef_L2, 3)
#model_train.adv_train_GAN(train_data, test_data)
adv_list = []
label_list = []
mag = .0
flag = 2

for i, data_batch in enumerate(train_data):
	feature, label = data_batch
	feature_temp = feature[:].cuda()
	feature, label = Variable(feature.cuda()), Variable(label.cuda())
	tot, loop, pred, pert_batch = model_train.batch_deepfool(netD, feature, label.data)
	#print(type(pert_batch))
	adv_list.append(pert_batch)
	label_list.append(label.data)
	if flag == 2:
		mag += torch.norm((feature_temp-pert_batch).view(64,-1), 2, 1).mean()
	else:
		mag += torch.max(torch.abs(feature_temp-pert_batch), 1)[0].mean()
	
print(1.0*mag/i)
adv_featureset = torch.cat(adv_list, 0)
labelset = torch.cat(label_list, 0)
res = TestAcc_tensor(netD,(adv_featureset, labelset))
print('Adv accuracy of netD: %.3f'%(res))

netD_robust = _netD_mnist()
netD_robust.cuda()
netD_robust.load_state_dict(torch.load('netD_L2_step1.pkl'))
res = TestAcc_tensor(netD_robust,(adv_featureset, labelset))
print('Adv accuracy of netD: %.3f'%(res))
torch.save((adv_featureset, labelset), path+'adv_deepfool.pt')
