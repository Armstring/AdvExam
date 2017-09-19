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
from utils.utils import accu,TestAcc, TestAdvAcc, AdcAcc_net
import models.model_train as model_train
from dataProcess.read_data import read_MNIST
import glob

#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)
batch_size = 64
test_batch_size = 1000
train_data , test_data = read_MNIST(batch_size, test_batch_size)


#netD = _netD_mnist()
#netD.cuda()
#netD.load_state_dict(torch.load('netD.pkl'))
#print('Test accuracy of netD: %.3f'%(TestAcc(netD,test_data)))
########################

#model_train.adv_train_gradient(train_data, test_data, 'sign', coef_FGSM, 1)
#model_train.adv_train_gradient(train_data, test_data, 'sign', coef_FGSM, 3)
#model_train.adv_train_gradient(train_data, test_data, 2, coef_L2, 1)
#model_train.adv_train_gradient(train_data, test_data, 2, coef_L2, 3)
model_train.adv_train_GAN(train_data, test_data)


'''
netD1 = _netD_mnist()
netD1.cuda()
netD1.load_state_dict(torch.load('netD_FGSM_step1.pkl'))
print('Test accuracy of netD_FGSM_step3: %.3f'%(TestAcc(netD1,test_data)))

netD2 = _netD_mnist()
netD2.cuda()
netD2.load_state_dict(torch.load('netD_gan_pert.pkl'))
print('Test accuracy of netD_gan_pert: %.3f'%(TestAcc(netD2,test_data)))

netD3 = _netD_mnist()
netD3.cuda()
netD3.load_state_dict(torch.load('netD_FGSM_step1.pkl'))
print('Test accuracy of netD_FGSM_step1: %.3f'%(TestAcc(netD3,test_data)))

#for filename in sorted(glob.glob('./adv_exam/*.pt')):
#	dataset_adv = torch.load(filename)
#	print(filename, TestAdvAcc(netD1, dataset_adv),  TestAdvAcc(netD2, dataset_adv), TestAdvAcc(netD3, dataset_adv))



######################## white-box
def testadvacc_whitebox(netD, flag, step_num, coef, dataset):

	print('gradient', flag, coef, step_num)
	mag = .0
	adv_list = []
	label_list = []
	for i,data_batch in enumerate(dataset):
		feature, label = data_batch
		#feature_temp = feature[:].cuda()
		perb_temp = model_train.advexam_gradient(netD, feature, label, flag, coef, step_num)
		if flag == 2:
			mag += torch.norm((feature_temp-perb_temp).view(64,-1), 2, 1).mean()
		else:
			mag += torch.max(torch.abs(feature_temp-perb_temp), 1)[0].mean()

		adv_list.append(perb_temp)
		label_list.append(label)

	print(1.0*mag/i)
	adv_featureset = torch.cat(adv_list, 0)
	labelset = torch.cat(label_list, 0)
	res = TestAdvAcc(netD,(adv_featureset, labelset))
	print('White-box Adv accuracy of netD: %.3f'%(res))

#netG = _netG_mnist()
#netG.cuda()
#netG.load_state_dict(torch.load('netG_gan_pert.pkl'))

ceof = 5.5
flag = 2
num_step = 5
testadvacc_whitebox(netD1, flag, num_step, ceof, test_data)
testadvacc_whitebox(netD2, flag, num_step, ceof, test_data)
testadvacc_whitebox(netD3, flag, num_step, ceof, test_data)
#print(AdcAcc_net(netD1, netG, test_data, ceof))
#print(AdcAcc_net(netD2, netG, test_data, ceof))
#print(AdcAcc_net(netD3, netG, test_data, ceof))


'''

