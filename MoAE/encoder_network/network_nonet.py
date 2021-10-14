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
class network_nonet(network):
    # -----------------------------------------------------------------------
    def __init__(self, device, model, hp, batch, verbose = False):
        super(network_nonet, self).__init__(device, hp, verbose)
        
        # no network
        self.model = model
        
        # Initialization
        # prepare parameter
        if self.hp.SHOULD_INIT_RANDOM:
            u_tensor = self.model.initialize_u_random(use_logit = False)
        else:
            u_tensor = self.model.initialize_u(batch = batch, use_logit = False)
        self.u = torch.nn.Parameter(u_tensor.to(self.device).detach().requires_grad_(True))
    
    # -----------------------------------------------------------------------
    def initialize_weights(self, batch):
        pass
        
    # -----------------------------------------------------------------------
    def get_u_range(self):
        text = " | u=[{:.4f},{:.4f}]".format(torch.min(self.u).cpu().detach().numpy(), torch.max(self.u).cpu().detach().numpy())
        return text
    
    # -----------------------------------------------------------------------
    def projection(self):
        with torch.no_grad():
            # u_tensor here is a "normalized" parameter, 0 represent the lowest value (can be negative), 1 represent the highest value
            # projection on u_tensor directly can return "negative value"
            # p_zero is the value that represent 0 in the parameter scale
    #                     u_tensor.copy_(torch.nn.functional.relu(u_tensor - self.model.p_zero()) + self.model.p_zero())
    #                     u_tensor.copy_(torch.sigmoid(u_tensor - self.model.p_zero()) + self.model.p_zero())
            self.u.copy_(torch.clamp(self.u - self.model.p_zero(), min=0.0, max=1.0) + self.model.p_zero())
        
    # -----------------------------------------------------------------------
    def forward(self, x):
        return self.u
    
    