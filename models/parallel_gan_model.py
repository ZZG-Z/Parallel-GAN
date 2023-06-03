import torch
from .base_model import BaseModel
from . import networks


class ParallelGANModel(BaseModel):
    """ This class implements the Parallel_GAN model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG para_GAN' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    Parallel GAN paper: https://ieeexplore.ieee.org/document/9864654
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For Parallel GAN, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the Parallel GAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.visual_names = ['fake_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if opt.net == 'translation':
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'feature', 'G_L1']
            self.recon_net = networks.define_recon_net().cuda()
            load_path = "./checkpoints/recon/latest_net_G_recon.pth" # the path of the trained recostruction model.
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.recon_net.load_state_dict(state_dict)
        elif opt.net == 'reconstruction':
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'recon_VGG', 'G_L1']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.net)
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.net == 'translation':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.net == 'reconstruction':
                self.optimizer_G = torch.optim.Adam([{'params':self.netG.module.decoder.parameters()}, 
                                                     {'params': self.netG.module.encoder.parameters(), 'lr': opt.lr / 10}],lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.net == 'reconstruction':
            self.fake_B = self.netG(self.real_B)[-1]
        elif self.opt.net == 'translation':
            self.trans_feature = self.netG(self.real_A)
            self.fake_B = self.trans_feature[-1]
            if self.isTrain:
                self.recon_feature = self.recon_net(self.real_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_feature = 0
        self.weight = [1/32, 1/16, 1/8, 1/4, 1/2]
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        if self.opt.net == 'translation':
            for i in range(len(self.trans_feature)-1):
                self.loss_feature += self.criterionL1(self.trans_feature[i], self.recon_feature[i]) * self.weight[i] * 10
            self.loss_G = self.loss_G_GAN + self.loss_feature + self.loss_G_L1
        elif self.opt.net == 'reconstruction':
            self.loss_recon_VGG = self.criterionVGG(self.fake_B, self.real_B) * 10
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_recon_VGG
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
