from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable

from models.model_train import advexam_gap, advexam_ll, advexam_gradient
from dataProcess.MNIST import read_MNIST
from nets.classifiers import _netD
from utils.utils import TestAcc, TestAdvAcc
#import torchvision.models as models


batch_size = 64
train_data , test_data = read_MNIST(batch_size, batch_size)

coef = 0.2
flag = 'sign'
path = "./adv_exam/"
loss_func = nn.CrossEntropyLoss()

#######
nc = 1
ndf = 16
netD = _netD()
netD.cuda()

netD.load_state_dict(torch.load('netD.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc(netD,test_data)))
#######
## least likely method does not really work!!

adv_list = []
label_list = []
for i,data_batch in enumerate(train_data):
	feature, label = data_batch
	feature,label = Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
	adv_list.append(advexam_gap(netD, feature, label, flag, coef, 1).data)
	label_list.append(label.data)

adv_featureset = torch.cat(adv_list, 0)
labelset = torch.cat(label_list, 0)
#if flag =='inf':
#	torch.save((adv_featureset, labelset), path+'adv_gap_inf_%.2f_iter.pt'%coef)
#elif flag=='sign':
#	torch.save((adv_featureset, labelset), path+'adv_gap_FGSM_%.2f_iter.pt'%coef)
#else:
#	torch.save((adv_featureset, labelset), path+'adv_gap_L%d_%.2f_iter.pt'%(flag, coef))

#dataset_adv = torch.load('./adv_exam/adv_FGSM_0.60.pt')
print('Adv accuracy of netD: %.3f'%(TestAdvAcc(netD,(adv_featureset, labelset))))



