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
import numpy as np
import torch
import torch.nn as nn
import enum

from .ssim import *

class loss_type(enum.Enum):
    L1 = 1
    L2 = 2
    SSIM = 3
    
class model_loss(nn.Module):
    def __init__(self, ssim_window_size = 5):
        super(model_loss, self).__init__()

        self.__loss_type = loss_type.L2
        self.mse = torch.nn.MSELoss()
        self.l1loss = torch.nn.L1Loss()
        self.ssim = SSIM3D(window_size = ssim_window_size)
        self.ssim_map = SSIM3D(window_size = ssim_window_size, size_average = False)
        
    def forward(self, prediction, target):
        if self.__loss_type == loss_type.L2:
            return self.mse(prediction, target)
        if self.__loss_type == loss_type.L1:
            return self.l1loss(prediction, target)
        if self.__loss_type == loss_type.SSIM:
            return 1.0 - self.ssim(prediction, target)
        
    def use_mse(self):
        self.__loss_type = loss_type.L2
        self.mse = torch.nn.MSELoss()
        
    def use_l1(self):
        self.__loss_type = loss_type.L1
        self.l1loss = torch.nn.L1Loss()
    
    def use_ssim(self, ssim_window_size = 5):
        self.__loss_type = loss_type.SSIM
        self.ssim = SSIM3D(window_size = ssim_window_size)
        self.ssim_map = SSIM3D(window_size = ssim_window_size, size_average = False)
        
    def _MSE(self,tensor1=None, tensor2=None, dim=None):
        error_type = "MSE"
        return torch.mean(torch.pow(tensor1 - tensor2, 2), dim=dim), error_type

    def _L1(self,tensor1=None, tensor2=None, dim=None):
        error_type = "L1"
        return torch.mean(torch.abs(tensor1 - tensor2), dim=dim), error_type

    def _SSIM(self, tensor1=None, tensor2=None, dim=None):
        error_type = "SSIM"
        return 1.0 - torch.mean(self.ssim_map(tensor1, tensor2), dim=dim), error_type

    def get_loss_map(self, prediction, target, dim):
        if self.__loss_type == loss_type.L2:
            return self._MSE(prediction, target, dim)
        if self.__loss_type == loss_type.L1:
            return self._L1(prediction, target, dim)
        if self.__loss_type == loss_type.SSIM:
            return self._SSIM(prediction, target, dim)

