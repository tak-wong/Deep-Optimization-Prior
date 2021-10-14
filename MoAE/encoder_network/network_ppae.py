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

from .network import network
from ..hyperparameter import *

# network of encoder
class network_ppae(network):
    # -----------------------------------------------------------------------
    def __init__(self, device, hp, verbose = False):
        super(network_ppae, self).__init__(device, hp, verbose)
        
        self.activation = nn.LeakyReLU()

        self.ConvLayer_ToOne = nn.Conv3d(2,10,(5,1,1), groups=2)
        self.ConvLayer_RealImag = nn.Conv3d(10,20,(10,1,1))
        self.ConvLayer_Both1 = nn.Conv3d(20,20,(15,1,1))
        self.ConvLayer_Both2 = nn.Conv3d(20,30,(31,1,1))

        self.BatchNorm10 = nn.BatchNorm3d(10)
        self.BatchNorm20_1 = nn.BatchNorm3d(20)
        self.BatchNorm20_2 = nn.BatchNorm3d(20)
        self.BatchNorm30 = nn.BatchNorm3d(30)
 
        self.FC_Last1 = nn.Linear(1020,5)
#         self.FC_Last1 = nn.Linear(1020,4)
        
        self.initialize_weights()
    
    # -----------------------------------------------------------------------
    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.ConvLayer_ToOne.weight)
        nn.init.kaiming_uniform_(self.ConvLayer_RealImag.weight)
        nn.init.kaiming_uniform_(self.ConvLayer_Both1.weight)
        nn.init.kaiming_uniform_(self.ConvLayer_Both2.weight)
        nn.init.kaiming_uniform_(self.FC_Last1.weight)

        nn.init.constant_(self.ConvLayer_ToOne.bias, 0.0) 
        nn.init.constant_(self.ConvLayer_RealImag.bias, 0.0)
        nn.init.constant_(self.ConvLayer_Both1.bias, 0.0)
        nn.init.constant_(self.ConvLayer_Both2.bias, 0.0)
        nn.init.constant_(self.FC_Last1.bias, 0.0)
        
    # -----------------------------------------------------------------------
    def forward(self, x):
        y = self.ConvLayer_ToOne(x)
        y = self.BatchNorm10(y)
        y = self.activation(y)
        
        y = self.ConvLayer_RealImag(y)
        y = self.BatchNorm20_1(y)
        y = self.activation(y)
        
        y = self.ConvLayer_Both1(y)
        y = self.BatchNorm20_2(y)
        y = self.activation(y)

        y = self.ConvLayer_Both2(y)
        y = self.BatchNorm30(y)
        y = self.activation(y)

        y = y.view(-1,1020)
        y = self.FC_Last1(y)

        y = self.activation(y)
        
        return y
