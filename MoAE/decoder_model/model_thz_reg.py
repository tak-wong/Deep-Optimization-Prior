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
import math
import os
import torch
import numpy as np
import scipy.io
import datetime

from PIL import Image
from torchvision import transforms

from .model_thz import *

class model_thz_reg(model_thz):
    # -----------------------------------------------------------------------
    def __init__(self, device, dataset, hp, SHOULD_PROJECT_BEFORE = False, verbose = True):
        super(model_thz_reg, self).__init__(device, dataset, SHOULD_PROJECT_BEFORE, verbose)
        
        self.noise_real = None
        self.noise_imag = None
        self.reg = None
        
        self.hp = hp
        
        if self.verbose:
            print("model_thz_reg.init")
            
    # -----------------------------------------------------------------------
    def get_name(self):
        return "model_thz_reg"
    
    # -----------------------------------------------------------------------
    def prepare_tensors(self, batch = None):
        # generate z-axis tensor
        z_reshape = self.z_np.reshape((1, 1, self.NZ, 1, 1))
        z_tile = np.tile(z_reshape, (1, 1, 1, self.NX, self.NY))
        self.z_tensor = torch.tensor(z_tile, requires_grad=False).float().to(self.device)
    
    # -----------------------------------------------------------------------
    def initialize_noise(self):
        # store to a single tensor
        noise_tensor = torch.zeros((1, 1, self.NZ, self.NX, self.NY), dtype=torch.float, requires_grad = False)

        return noise_tensor
    
    # -----------------------------------------------------------------------
    def get_p(self):
        params = {
            "sigma" : self.sigma,
            "mu" : self.mu,
            "ehat" : self.ehat,
            "phi" : self.phi,
        }
            
        if (self.noise_real is not None):
            params["noise_real"] = self.noise_real
            
        if (self.noise_imag is not None):
            params["noise_imag"] = self.noise_imag            
        
        return params
            
    # -----------------------------------------------------------------------
    def forward_reg(self, batch_shape, batch_predict, noise_real, noise_imag):
        # this is the forward model, as a decoder
        model_predict = torch.zeros(batch_shape, dtype=torch.float, requires_grad=True, device=self.device).clone()
        
        reg = torch.zeros((2), dtype=torch.float, requires_grad=True, device=self.device).clone()
        
        """
        Model based decoder
        """
        sigma_norm = batch_predict[:, :, 0, :, :]
        mu_norm = batch_predict[:, :, 1, :, :]
        ehat_norm = batch_predict[:, :, 2, :, :]
        phic_norm = batch_predict[:, :, 3, :, :]
        phis_norm = batch_predict[:, :, 4, :, :]

        # do projection
        if (self.SHOULD_PROJECT_BEFORE):
            sigma_proj, mu_proj, ehat_proj, phic_proj, phis_proj = self.projection(sigma_norm, mu_norm, ehat_norm, phic_norm, phis_norm)
        else:
            # extend all parameters in z-axis
            sigma_proj = sigma_norm
            mu_proj = mu_norm       
            ehat_proj = ehat_norm              
            phic_proj = phic_norm
            phis_proj = phis_norm

        # calculate all predicted parameters
        sigma_one, mu_one, ehat_one, phic_one, phis_one = self.scaling_unfold(sigma_proj, mu_proj, ehat_proj, phic_proj, phis_proj)
        
        # fold to phage angle
        phi_one = torch.atan2( phis_one, phic_one )

        # extend all parameters in z-axis
        sigma = sigma_one.unsqueeze(2).repeat(1, 1, self.NZ, 1, 1)
        mu = mu_one.unsqueeze(2).repeat(1, 1, self.NZ, 1, 1)
        ehat = ehat_one.unsqueeze(2).repeat(1, 1, self.NZ, 1, 1)
        phi = phi_one.unsqueeze(2).repeat(1, 1, self.NZ, 1, 1)
        
        # calculate sinc(x) = sin(pi * x) / (pi * x)
        pi_sigma_mu = math.pi * sigma * (self.z_tensor - mu)
        sin_pi_sigma_mu = torch.sin( math.pi * sigma * (self.z_tensor - mu) )
        sin_pi_sigma_mu[ torch.abs(pi_sigma_mu) < self.FLT_EPSILON ] = 1
        pi_sigma_mu[ torch.abs(pi_sigma_mu) < self.FLT_EPSILON ] = 1
        sinc_sigma_mu = (sin_pi_sigma_mu) / (pi_sigma_mu)

        # calculate phase item
        omega_z_phi = self.OMEGA * self.z_tensor - phi

        # compute the model
        if noise_real is None or noise_imag is None:
            model_predict[:, self.INDEX_REAL, :, :, :] =      ehat * sinc_sigma_mu * torch.cos(omega_z_phi)
            model_predict[:, self.INDEX_IMAG, :, :, :] = -1 * ehat * sinc_sigma_mu * torch.sin(omega_z_phi)
        else:
            model_predict[:, self.INDEX_REAL, :, :, :] =      ehat * sinc_sigma_mu * torch.cos(omega_z_phi) + self.hp.ALPHA_S * noise_real
            model_predict[:, self.INDEX_IMAG, :, :, :] = -1 * (ehat * sinc_sigma_mu * torch.sin(omega_z_phi) + self.hp.ALPHA_S * noise_imag)
        
        # regularization
        if self.hp.ALPHA_S > 0.0:
            noise_reg = self.hp.ALPHA_S * self.hp.LAMBDA_S * (self.Lq_norm(noise_real, self.hp.NORM_Q, dim=(0, 1, 2)) + self.Lq_norm(noise_imag, self.hp.NORM_Q, dim=(0, 1, 2)))
            reg[0] = torch.mean( noise_reg )
        else:
            reg[0] = 0.0
        
        
        reg[1] = 0.0
        if np.count_nonzero(self.hp.LAMBDA_I) > 0:
            if self.hp.LAMBDA_I[0] > 0.0:
                ehat_grad = torch.mean(torch.abs(ehat[:, :, :, 0:self.NX-1, :] - ehat[:, :, :, 1:self.NX, :])) + torch.mean(torch.abs(ehat[:, :, :, :, 0:self.NY-1] - ehat[:, :, :, :, 1:self.NY]))
                reg[1] += np.square(self.hp.LAMBDA_I[0] * self.MAX_VALUE) * ehat_grad
                
            if self.hp.LAMBDA_I[1] > 0.0:
                sigma_grad = torch.mean(torch.abs(sigma[:, :, :, 0:self.NX-1, :] - sigma[:, :, :, 1:self.NX, :])) + torch.mean(torch.abs(sigma[:, :, :, :, 0:self.NY-1] - sigma[:, :, :, :, 1:self.NY]))
                reg[1] += np.square(self.hp.LAMBDA_I[1] * self.MAX_VALUE) * sigma_grad
                
            if self.hp.LAMBDA_I[2] > 0.0:
                mu_grad = torch.mean(torch.abs(mu[:, :, :, 0:self.NX-1, :] - mu[:, :, :, 1:self.NX, :])) + torch.mean(torch.abs(mu[:, :, :, :, 0:self.NY-1] - mu[:, :, :, :, 1:self.NY]))
                reg[1] += np.square(self.hp.LAMBDA_I[2] * self.MAX_VALUE) * mu_grad
                
            if self.hp.LAMBDA_I[3] > 0.0:
                phic_grad = torch.mean(torch.abs(torch.cos(phi[:, :, :, 0:self.NX-1, :]) - torch.cos(phi[:, :, :, 1:self.NX, :]))) + torch.mean(torch.abs(torch.cos(phi[:, :, :, :, 0:self.NY-1]) - torch.cos(phi[:, :, :, :, 1:self.NY])))
                reg[1] += np.square(self.hp.LAMBDA_I[3] * self.MAX_VALUE) * phic_grad
                
            if self.hp.LAMBDA_I[4] > 0.0:
                phis_grad = torch.mean(torch.abs(torch.sin(phi[:, :, :, 0:self.NX-1, :]) - torch.sin(phi[:, :, :, 1:self.NX, :]))) + torch.mean(torch.abs(torch.sin(phi[:, :, :, :, 0:self.NY-1]) - torch.sin(phi[:, :, :, :, 1:self.NY])))
                reg[1] += np.square(self.hp.LAMBDA_I[4] * self.MAX_VALUE) * phis_grad
    
        with torch.no_grad():
            self.sigma = sigma_one.clone().requires_grad_(False).squeeze_()
            self.mu = mu_one.clone().requires_grad_(False).squeeze_()
            self.ehat = ehat_one.clone().requires_grad_(False).squeeze_()
            self.phi = phi_one.clone().requires_grad_(False).squeeze_()
            if noise_real is None or noise_imag is None:
                self.noise_real = None
                self.noise_imag = None
            else:
                self.noise_real = torch.sum(noise_real, dim=(0, 1, 2), keepdim=False).clone().requires_grad_(False)
                self.noise_imag = torch.sum(noise_imag, dim=(0, 1, 2), keepdim=False).clone().requires_grad_(False)
            
        return model_predict, reg
    
    # -----------------------------------------------------------------------
    def save_mat(self, path_file, sigma = None, mu = None, ehat = None, phi = None, noise_real = None, noise_imag = None, loss_map = None):
        if (self.verbose):
            print("Writing mat File")
            
        if sigma is None:
            sigma = self.sigma

        if mu is None:
            mu = self.mu
            
        if ehat is None:
            ehat = self.ehat
            
        if phi is None:
            phi = self.phi
            
        if noise_real is None:
            noise_real = self.noise_real
            
        if noise_imag is None:
            noise_imag = self.noise_imag
            
        matfiledata = {} # make a dictionary to store the MAT data in
        matfiledata[u'Ehat'] = ehat.cpu().detach().numpy()
        matfiledata[u'Sigma'] = sigma.cpu().detach().numpy()
        matfiledata[u'Mu'] = mu.cpu().detach().numpy()
        matfiledata[u'Phi'] = phi.cpu().detach().numpy()
            
        if (loss_map is not None):
            matfiledata[u'Loss'] = loss_map.cpu().detach().numpy()
            
        if (noise_real is not None):
            matfiledata[u'Noise_real'] = noise_real.cpu().detach().numpy()
            
        if (noise_imag is not None):
            matfiledata[u'Noise_imag'] = noise_imag.cpu().detach().numpy()

        scipy.io.savemat(path_file, matfiledata)

        if (self.verbose):
            print("{} is saved".format(path_file))
            
        return path_file
    