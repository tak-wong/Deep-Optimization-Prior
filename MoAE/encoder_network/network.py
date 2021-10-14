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
import torch.nn as nn

from .unet.skip import skip
from ..hyperparameter import *

def get_network_encoder_unet(device, model, hp):
    network =  skip(num_input_channels  = model.NQ, 
                    num_output_channels = model.NP,
                    num_channels_down   = hp.NET_UNET_CHANNELS_DOWN,
                    num_channels_up     = hp.NET_UNET_CHANNELS_UP,
                    num_channels_skip   = hp.NET_UNET_CHANNELS_SKIP,
                    upsample_mode       = hp.NET_UNET_MODE,
                    need_sigmoid        = hp.NET_UNET_NEED_SIGMOID, 
                    need_bias           = hp.NET_UNET_NEED_BIAS, 
                    pad                 = hp.NET_UNET_PAD, 
                    act_fun             = hp.NET_UNET_ACT_FUNC).type(torch.cuda.FloatTensor)
    
    return network

# network of encoder
class network(nn.Module):
    # -----------------------------------------------------------------------
    def __init__(self, device, hp, verbose = False):
        super(network, self).__init__()
        self.verbose = verbose

        self.device = device
        self.hp = hp

        if self.verbose:
            print("network_encoder is initialized")
    
    # -----------------------------------------------------------------------
    def initialize_weights(self):
        pass
        
    # -----------------------------------------------------------------------
    def forward(self, x):
        pass
