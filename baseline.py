from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from nets.classifiers import _netD_mnist
from constants import *
from utils.utils import accu,TestAcc, TestAdvAcc, AdcAcc_net
from models.model_train import advexam_gap, advexam_ll, advexam_gradient
from dataProcess.read_data import read_MNIST


#torch.manual_seed(31415926)
#torch.cuda.manual_seed(31415926)
batch_size = 64
test_batch_size = 1000
train_data , test_data = read_MNIST(batch_size, test_batch_size)


netD = _netD_mnist()
netD.cuda()
loss_func = nn.CrossEntropyLoss()

'''
##### Train netD on real data
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))

for epoch in range(epoch_num):
  running_loss = .0
  running_acc = .0

  for i,data_batch in enumerate(train_data):
    netD.zero_grad()
    feature,label = data_batch
    feature_v, label_v = Variable(feature.cuda()), Variable(label.cuda())
    outputs_v = netD(feature_v)
    loss = loss_func(outputs_v, label_v)
    _, pred = torch.max(outputs_v, 1)
    loss.backward()
    optimizerD.step()
    running_loss += loss.data[0]
    running_acc += accu(outputs_v, label_v)
    if i % 200 ==199:
      print('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i + 1,
      	running_loss / 200, running_acc/200))
      running_loss = .0
      running_acc = .0
  print('Test accuracy of netD: %.3f'%(TestAcc(netD,test_data)))

print('Finished Pre_train netD')
torch.save(netD.state_dict(), './netD.pkl')

#netD.load_state_dict(torch.load('netD.pkl'))
print('Test accuracy of netD: %.3f'%(TestAcc(netD,test_data)))
'''
######################
## gap base
adv_list = []
label_list = []
flag = 2
step_num = 15
path = "./adv_exam/"
coef = 1.5
print('gradient', flag, coef, step_num)
mag = .0

for i,data_batch in enumerate(train_data):
  feature, label = data_batch
  feature_temp = feature[:].cuda()
  perb_temp = advexam_gradient(netD, feature, label, flag, coef, step_num)
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
print('Adv accuracy of netD: %.3f'%(res))

if flag=='sign':
  torch.save((adv_featureset, labelset), path+'adv_gradient_FGSM_step%d.pt'%(step_num))
else:
  torch.save((adv_featureset, labelset), path+'adv_gradient_L%d_step%d.pt'%(flag, step_num))



