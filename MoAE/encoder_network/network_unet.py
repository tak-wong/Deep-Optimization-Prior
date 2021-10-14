#########################################################################################################################
##   Distribution code Version 1.0 -- 14/10/2021 by Tak Ming Wong Copyright 2021, University of Siegen
##
##   The Code is created based on the method described in the following paper 
##   [1] "Deep Optimization Prior for THz Model Parameter Estimation", T.M. Wong, H. Bauermeister, M. Kahl, P. Haring Bolivar, M. Moeller, A. Kolb, 
##   Winter Conference on Applications of Computer Vision (WACV) 2022.
##  
##   If you use this code in your scientific publication, please cite the mentioned paper.
##   The code and the algorithm are for non-comercial use only.
##
##   For other details, please visit website https://github.com/tak-wong/Deep-Optimization-Prior
#########################################################################################################################
import torch

from .unet.skip import skip
from .network import network
from ..hyperparameter import *


# network of encoder
class network_unet(network):
    # -----------------------------------------------------------------------
    def __init__(self, device, model, hp, verbose = False):
        super(network_unet, self).__init__(device, hp, verbose)
        
        # Architecture 1
        self.unet =  skip(num_input_channels  = model.NQ, 
                          num_output_channels = model.NP,
                          num_channels_down   = hp.NET_UNET_CHANNELS_DOWN,
                          num_channels_up     = hp.NET_UNET_CHANNELS_UP,
                          num_channels_skip   = hp.NET_UNET_CHANNELS_SKIP,
                          upsample_mode       = hp.NET_UNET_MODE,
                          need_sigmoid        = hp.NET_UNET_NEED_SIGMOID, 
                          need_bias           = hp.NET_UNET_NEED_BIAS, 
                          pad                 = hp.NET_UNET_PAD, 
                          act_fun             = hp.NET_UNET_ACT_FUNC).type(torch.cuda.FloatTensor)

#         # Architecture 2        
#         self.unet =  skip(num_input_channels  = model.NQ, 
#                           num_output_channels = model.NQ,
#                           num_channels_down   = hp.NET_UNET_CHANNELS_DOWN,
#                           num_channels_up     = hp.NET_UNET_CHANNELS_UP,
#                           num_channels_skip   = hp.NET_UNET_CHANNELS_SKIP,
#                           upsample_mode       = hp.NET_UNET_MODE,
#                           need_sigmoid        = hp.NET_UNET_NEED_SIGMOID, 
#                           need_bias           = hp.NET_UNET_NEED_BIAS, 
#                           pad                 = hp.NET_UNET_PAD, 
#                           act_fun             = hp.NET_UNET_ACT_FUNC).type(torch.cuda.FloatTensor)
        
#         self.conv = torch.nn.Conv2d(model.NQ, model.NP, kernel_size=1, bias=hp.NET_UNET_NEED_BIAS, padding_mode=hp.NET_UNET_PAD)
    
    # -----------------------------------------------------------------------
    def initialize_weights(self):
        for m in self.unet.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
#                 torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
    # -----------------------------------------------------------------------
    def forward(self, x):
        # Architecture 1
        y = self.unet(x)
        
        return y