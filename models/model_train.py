# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from constants import *
from utils.utils import accu,TestAcc_dataloader, TestAcc_tensor
import numpy as np
from nets.classifiers import _netD_mnist, _netG_mnist
import copy
from torch.autograd.gradcheck import zero_gradients


def infnorm(tensor):
	return torch.max(torch.abs(tensor))

def advexam_gradient(netD, feature, label, flag, coef, iter_num):
	perturb = torch.zeros(feature.size()).cuda()
	#perturb.normal_(0.0,0.0001)
	feature, label = Variable(feature.cuda()), Variable(label.cuda())
	perturb = Variable(perturb, requires_grad = True)
	
	p_coef = 0.0
	if iter_num==1:
		p_coef = coef
	else:
		p_coef = 1.1*coef/iter_num

	for i in range(iter_num):
		feature_adv = feature + perturb
		feature_adv.data.clamp_(min=-1.0, max = 1.0)
		outputs = netD(feature_adv)
		error = .0
		for j in range(outputs.size()[0]):
			error -= outputs[j][label.data[j]]
		error.backward()
		_, pred = torch.max(outputs, 1)

		for j,image in enumerate(feature):
			if pred[j]==label[j]:
				if flag=='sign':
					perturb[j] +=  p_coef*torch.sign(perturb.grad[j])
				else:
					duel_norm = flag/(flag-1.0)
					perturb[j] += p_coef * perturb.grad[j]/torch.norm(perturb.grad[j].data, p = duel_norm)
		
	feature_adv = feature + perturb
	feature_adv.data.clamp_(min=-1.0, max = 1.0)
	return feature_adv.data

def advexam_ll(netD, feature, label, flag, coef, iter_num):
	perturb = torch.FloatTensor(feature.size()).cuda()
	perturb.normal_(0.0,0.0001)
	feature, label = Variable(feature.cuda()), Variable(label.cuda())
	perturb = Variable(perturb, requires_grad = True)
	
	p_coef = 0.0
	if iter_num==1:
		p_coef = coef
	else:
		p_coef = 1.1*coef/iter_num

	for i in range(iter_num):
		feature_adv = feature + perturb
		feature_adv.data.clamp_(min=-1.0, max = 1.0)
		outputs = netD(feature_adv)
		error = torch.min(outputs, 1)[0].sum()
		error.backward()
		_, pred = torch.max(outputs, 1)

		for j,image in enumerate(feature):
			if pred[j]==label[j]:
				if flag=='sign':
					perturb[j] +=  p_coef*torch.sign(perturb.grad[j])
				else:
					duel_norm = flag/(flag-1.0)
					perturb[j] += p_coef * perturb.grad[j]/torch.norm(perturb.grad[j].data, p = duel_norm)
		
	feature_adv = feature + perturb
	feature_adv.data.clamp_(min=-1.0, max = 1.0)
	return feature_adv.data

def advexam_gap(netD, feature, label, flag, coef, iter_num):
	perturb = torch.FloatTensor(feature.size()).cuda()
	perturb.normal_(0.0,0.0001)
	feature, label = Variable(feature.cuda()), Variable(label.cuda())
	perturb = Variable(perturb, requires_grad = True)
	
	p_coef = 0.0
	if iter_num==1:
		p_coef = coef
	else:
		p_coef = 1.1*coef/iter_num

	for i in range(iter_num):
		feature_adv = feature + perturb
		feature_adv.data.clamp_(min=-1.0, max = 1.0)
		outputs = netD(feature_adv)
		top2 = outputs.topk(2,1)[0]
		error = -(top2[:,0] - top2[:,1]).sum()#
		error.backward()
		_, pred = torch.max(outputs, 1)

		for j,image in enumerate(feature):
			if pred[j]==label[j]:
				if flag=='sign':
					perturb[j] +=  p_coef*torch.sign(perturb.grad[j])
				else:
					duel_norm = flag/(flag-1.0)
					perturb[j] += p_coef * perturb.grad[j]/torch.norm(perturb.grad[j].data, p = duel_norm)
		
	feature_adv = feature + perturb
	feature_adv.data.clamp_(min=-1.0, max = 1.0)
	return feature_adv.data

def adv_train_gan_G(netG, netD, loss_func, feature, perturb, label, coef, optimizerG):
	
	#G_input = torch.FloatTensor(label.size()[0], nz_netG).cuda()
	#G_input.normal_(0,1)
	#label_onehot = torch.zeros(label.size()[0], 10).cuda()
	#label_onehot.scatter_(1, label.data.view(-1,1), 1)
	#D_info = torch.cat([label_onehot.cuda(), G_input], 1)
	#D_info = Variable(D_info)
	#G_input = Variable(G_input, requires_grad = True)
	#indd = torch.mm(label.cpu().view(-1,1), torch.autograd.Variable(torch.ones(1, ngf_netG).long())).cuda()
	#indd = indd.view(-1,1,ngf_netG)
	for j in range(3):
		netG.zero_grad()
		adv_perb = netG(feature)

		adv_perb_vec = adv_perb.view(adv_perb.size()[0], 1, -1)
		perturb_vec = perturb.view(perturb.size()[0], -1 ,1)
		loss1 = -0.1*torch.sum(torch.clamp(torch.bmm(adv_perb_vec, perturb_vec), max = 0.0))


		fake = feature + coef*adv_perb
		fake.data.clamp_(min=-1.0, max = 1.0)

		outputs = netD(fake)
		#errorG = -loss_func(outputs, label)
		errorG = .0
		for j in range(outputs.size()[0]):
			errorG += outputs[j][label.data[j]]

		loss = loss1 + errorG
		loss.backward()
		
		optimizerG.step()
	adv_perb = netG(feature)#, indd, G_input)
	fake = feature + coef*adv_perb
	outputs = netD(fake)
	errorG = -loss_func(outputs, label)
	accG = accu(outputs, label)/batch_size
	return (errorG, accG, netG, fake)

def adv_train_gan_D(netD, loss_func, fake, feature, label, optimizerD):
	
	for i in range(1):
		netD.zero_grad()
		outputs = netD(feature)
		error_real = loss_func(outputs, label)
		acc_real = accu(outputs, label)/batch_size
		error_real.backward()

		outputs = netD(fake.detach())
		error_fake = loss_func(outputs, label)
		acc_fake = accu(outputs, label)/batch_size
		error_fake.backward()

		optimizerD.step()
	#for p in netD.parameters():
		#p.data.clamp_(-0.02, 0.02)
	return (error_real, acc_real, error_fake, acc_fake, netD)


def adv_train_GAN(train_data, test_data):
	coef = coef_gan
	netD = _netD_mnist()
	netD.cuda()
	netG = _netG_mnist()
	netG.cuda()
	
	#optimizerD = optim.SGD(netD.parameters(), lr=lr_D)
	optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))
	#optimizerG = optim.SGD(netG.parameters(), lr=lr_G)
	optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.9, 0.999))
	loss_func = nn.CrossEntropyLoss()

	for epoch in range(epoch_num):
		running_acc_D = .0
		running_loss_D = .0

		for i,data_batch in enumerate(train_data):
			feature, label = data_batch
			feature, label= Variable(feature.cuda(), requires_grad = True), Variable(label.cuda())
			outputs = netD(feature)
			error = loss_func(outputs, label)
			error.backward()
			perturb = feature.grad
			#perturb = torch.sign(perturb)
			perturb.volatile = False
			perturb.requires_grad = True
			#print(perturb.requires_grad, perturb.volatile)

			(errorG, accG, netG, fake) = adv_train_gan_G(netG, netD, loss_func, feature, perturb, label, coef, optimizerG)
			(error_real, acc_real, error_fake, acc_fake, netD) = adv_train_gan_D(netD, loss_func, fake, feature, label, optimizerD)
			if i%200==0:
				print('[%d/%d][%d/%d]: Acc_fake: %.3f; Acc_real: %.3f;'
					%(epoch, epoch_num, i, len(train_data), acc_fake, acc_real))
			if epoch%5==2 and i%500==0:
					vutils.save_image(fake.data, './adv_image/adv_image_epoch_%03d_%03d.png' %(epoch,i), normalize = True)
			
		if epoch % 2 == 0:
			dataset_adv = torch.load('./adv_exam/adv_gradient_L2_step1.pt')
			test_acc = TestAcc_dataloader(netD, test_data)
			test_adv_acc = TestAcc_tensor(netD, dataset_adv)
			print('[%d/%d]Test accu: %.3f' %(epoch, epoch_num, test_acc) )
			print('[%d/%d]Test ADV accu: %.3f' %(epoch, epoch_num, test_adv_acc) )
	torch.save(netD.state_dict(), './netD_gan.pkl')
	torch.save(netG.state_dict(), './netG_gan.pkl')

def adv_train_gradient(train_data, test_data, norm, coef, adv_step):
	netD = _netD_mnist()
	netD.cuda()
	optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))
	loss_func = nn.CrossEntropyLoss()

	if True:
		print('Start adv Training with ' + str(norm) + ' adv example! Step:' + str(adv_step))
		for epoch in range(epoch_num):
			running_loss_D = .0
			running_acc_D = .0
			for i, data_batch in enumerate(train_data):
				netD.zero_grad()
				feature,label = data_batch
				netD_cp = copy.deepcopy(netD)
				feature_adv = advexam_gradient(netD_cp, feature, label, norm, coef, adv_step)
				feature_adv, label = Variable(feature_adv), Variable(label.cuda())
				
				outputs = netD(feature_adv.detach())
				errorD_real = loss_func(outputs, label)
				errorD_real.backward()
				optimizerD.step()

				running_loss_D += errorD_real
				running_acc_D += accu(outputs, label)/batch_size

				if i%50==49:
					print('[%d/%d][%d/%d] Adv perf: %.4f / %.4f'
						% (epoch, epoch_num, i, len(train_data),
							running_loss_D.data.cpu().numpy()[0]/50, running_acc_D/50))
					running_loss_D = .0
					running_acc_D = .0
				if epoch%5==2 and i%500==0:
					vutils.save_image(feature_adv.data, './adv_image/adv_image_epoch_%03d_%03d.png' %(epoch,i), normalize = True)
			if epoch % 3 == 0:
				dataset_adv = torch.load('./adv_exam/adv_gradient_FGSM_step1.pt')
				test_acc = TestAcc_dataloader(netD, test_data)
				test_adv_acc = TestAcc_tensor(netD, dataset_adv)
				print('[%d/%d]Test accu: %.3f' %(epoch, epoch_num, test_acc) )
				print('[%d/%d]Test ADV accu: %.3f' %(epoch, epoch_num, test_adv_acc) )
			if epoch % 13 ==12:
				optimizerD.param_groups[0]['lr'] /= 2.0
		if norm=='sign':
			torch.save(netD.state_dict(), './netD_FGSM_step%d.pkl' %(adv_step))
		else:
			torch.save(netD.state_dict(), './netD_L%1d_step%d.pkl' %(norm,adv_step))
 


def batch_deepfool(net, cur_img, label, num_classes=10, overshoot=0.0, max_iter=20,t_p=0.3):

    f_image = net.forward(cur_img)
    batch_size = cur_img.size(0)
    I = torch.sort(f_image,1)[1].data
    nv_idx = torch.arange(I.size(1)-1, -1, -1).long().cuda()
    I = I.index_select(1, nv_idx)

    #I = I[:,0:num_classes]
    #label = I[:,0]

    input_shape = cur_img.size()
    x = torch.autograd.Variable(cur_img.data,requires_grad = True)

    pert_image = torch.zeros(input_shape).cuda()
    w = torch.zeros(input_shape).cuda()
    r_tot = torch.zeros(input_shape).cuda()
    pert = torch.FloatTensor((np.inf,)*batch_size).cuda()

    loop_i = 0

    #x = pert_image
    fs = net.forward(x)
    
    
    fs_list = [fs[i,I[i,k]] for k in range(num_classes) for i in range(batch_size)]
    k_i = I[:,0]
    truth_percent = torch.sum(torch.eq(k_i,label))/float(batch_size)

    while truth_percent>t_p and loop_i < max_iter:

        truth_guards = torch.eq(k_i,label)
        index_truth = [i for i in range(batch_size) if truth_guards[i] == 1]
        
        fs_backer = [fs[i,I[i,0]] for i in index_truth]
        fs_backer = torch.sum(torch.stack(tuple(fs_backer),0))
        fs_backer.backward(retain_variables=True)

        grad_orig = torch.Tensor(x.grad.data.cpu()).cuda()

        for k in range(1, num_classes):
            zero_gradients(x)
            fs_backer = [fs[i,I[i,k]] for i in index_truth]
            fs_backer = torch.sum(torch.stack(tuple(fs_backer),0))
            fs_backer.backward(retain_variables=True)
            cur_grad = torch.Tensor(x.grad.data.cpu()).cuda()

            # set new w_k and new f_k
            # set new w_k and new f_k
            r_i = torch.zeros(input_shape).cuda()
            pert_k = torch.zeros(batch_size).cuda()
            f_k = [0]*batch_size
            f_k_batch = [0]*batch_size
            w_k = cur_grad - grad_orig
            w_k_batch = [0]*batch_size
            for i in index_truth:
                f_k[i] = fs[i,I[i,k]] -fs[i,I[i,0]]
                f_k_batch[i] = torch.abs(f_k[i].data)
                w_k_batch[i] = torch.norm(w_k[i]) + 0.000001
                pert_k[i] = (f_k_batch[i]/w_k_batch[i])[0]
                if pert_k[i] <= pert[i]:
                    pert[i] = pert_k[i]
                    w[i] = w_k[i]
                r_i[i] =  pert[i]*w[i]/w_k_batch[i]
        
        r_tot = r_tot + r_i
        pert_image =cur_img.data + (1+overshoot)*r_tot
        pert_image.clamp_(min=-1.0, max = 1.0)
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = torch.sort(fs,1)[1].data
        nv_idx = torch.arange(k_i.size(1)-1, -1, -1).long().cuda()
        k_i = k_i.index_select(1, nv_idx)
        k_i = k_i[:,0]
        truth_percent = torch.sum(torch.eq(k_i,label))/float(batch_size)
        loop_i += 1

    print(loop_i, truth_percent)
    r_tot = (1+overshoot)*r_tot

    return torch.mean(r_tot,0), loop_i, k_i, pert_image


























