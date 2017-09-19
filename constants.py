####################################
#data parameters
image_size = 28*28
image_shape = (28, 28)

####################################
#network parameters
nc_netG = 1
ndf_netG = 16
#ngf_netG = 256
#nz_netG =128
#ninput_netG = 8*ndf_netG + nz_netG


nc_netD = 1
ndf_netD = 256

####################################
#training parameters
epoch_num = 18
lr_D = 0.001
lr_G = 0.0005

###################################
###perturbation magnitude for training
coef_FGSM = 0.2
coef_L2 = 6.2

coef_FGSM_gap = 0.3
coef_L2_gap = 3.0

coef_gan = 0.47