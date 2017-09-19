# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from constants import *

def weights_init(layer):
  classname = layer.__class__.__name__
  if classname.find('Conv') != -1:
    layer.weight.data.normal_(0.0, 0.07)
  elif classname.find('Lin') != -1:
    layer.weight.data.normal_(0.0, 0.07)

def accu(output, label):
  _, pred = torch.max(output, 1)
  return 1.0*(pred.data==label.data).sum()/label.size()[0]

def TestAcc(net, dataset):
  adv_acc = .0
  num = 0
  net.eval()
  for i,data_batch in enumerate(dataset):
    feature, label = data_batch
    feature, label = Variable(feature.cuda()), Variable(label.cuda())
    outputs = net(feature)
    adv_acc += accu(outputs, label) * label.size()[0]
    num += label.size()[0]
  return 1.0*adv_acc/num

def TestAdvAcc(net, dataset):
  adv_acc = .0
  adv_feature, label = dataset
  adv_dataset = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(adv_feature, label),
    batch_size=64, shuffle=True, drop_last = False)
  num = 0
  for i,data_batch in enumerate(adv_dataset):
    adv_feature, label  = data_batch    
    adv_feature, label  = Variable(adv_feature.cuda()), Variable(label.cuda()) 
    outputs = net(adv_feature)
    adv_acc += accu(outputs, label)* label.size()[0]
    num += label.size()[0]
  return 1.0*adv_acc/num

def AdvAcc_exam(netD, advpath):
  #path = './adv_exam/adv_FGSM_0.30.pt'
  dataset_adv = torch.load(advpath)
  test_adv_acc = TestAdvAcc(netD, dataset_adv)
  return test_adv_acc

def AdcAcc_net(netD, netG, testset, coef):
  adv_acc = .0
  #nz = 128
  #batch_size = 64
  #G_input = torch.FloatTensor(batch_size, nz).cuda()
  loss_func = nn.CrossEntropyLoss()
  for i,data_batch in enumerate(testset):
    #G_input.normal_(0,1)
    feature, label = data_batch
    feature, label = Variable(feature.cuda(),requires_grad = True), Variable(label.cuda())
    
    #G_input_var = Variable(G_input)
    #indd = torch.mm(label.cpu().view(-1,1), torch.autograd.Variable(torch.ones(1, ngf_netG).long())).cuda()
    #indd = indd.view(-1,1,ngf_netG)

    outputs = netD(feature)
    error = loss_func(outputs, label)
    error.backward()
    perturb = feature.grad
    perturb = torch.sign(perturb)
    

    adv_perb = netG(feature)
    fake = feature + coef*adv_perb
    outputs = netD(fake)
    adv_acc += accu(outputs, label)
  return adv_acc/(i+1)












