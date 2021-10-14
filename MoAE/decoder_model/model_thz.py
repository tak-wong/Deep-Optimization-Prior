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

from .model import *

# THz model with single layer reflection
class model_thz(model):
    # -----------------------------------------------------------------------
    def __init__(self, device, dataset, SHOULD_PROJECT_BEFORE = False, verbose = True):
        super(model_thz, self).__init__(verbose)
        
        self.device = device
        
        self.EPS = 1e-10
        
        self.z_np = dataset.z_np
        self.NC = dataset.NC
        self.NZ = dataset.NZ
        self.NX = dataset.NX
        self.NY = dataset.NY
        self.INDEX_REAL = dataset.INDEX_REAL
        self.INDEX_IMAG = dataset.INDEX_IMAG
        self.OMEGA = dataset.OMEGA
        self.ZERO_PADDING = dataset.ZERO_PADDING
        self.MAX_VALUE = dataset.MAX_VALUE
        
        self.SHOULD_PROJECT_BEFORE = SHOULD_PROJECT_BEFORE

        self.Z_MIN = dataset.z_np[0]
        self.Z_MAX = dataset.z_np[-1]

        self.FLT_EPSILON = 1e-8

        # number of data channels
        self.NQ = self.NC * self.NZ
        
        # number of parameters
        self.NP = 5
        
        # define parameters
        self.sigma = None
        self.mu = None
        self.ehat = None
        self.phi = None
        
        if self.verbose:
            print("model_thz.init")
            
    # -----------------------------------------------------------------------
    def get_name(self):
        return "model_thz"
            
    # -----------------------------------------------------------------------
    def get_p(self):
        params = {
            "sigma" : self.sigma,
            "mu" : self.mu,
            "ehat" : self.ehat,
            "phi" : self.phi,
        }

        return params
    
    # -----------------------------------------------------------------------
    def prepare_tensors(self, batch = None):
        # generate z-axis tensor
        z_reshape = self.z_np.reshape((1, 1, self.NZ, 1, 1))
        z_tile = np.tile(z_reshape, (1, 1, 1, self.NX, self.NY))
        self.z_tensor = torch.tensor(z_tile, requires_grad=False).float().to(self.device)
        
    # -----------------------------------------------------------------------
    def prepare_tensors_perpixel(self):
        # generate z-axis tensor
        z_reshape = self.z_np.reshape((1, 1, self.NZ, 1, 1))
        self.z_tensor = torch.tensor(z_reshape, requires_grad=False).float().to(self.device)
            
    # -----------------------------------------------------------------------
    def scaling(self, sigma_norm, mu_norm, ehat_norm, phi_norm):
        # To scale up all normalized parameters
        sigma = sigma_norm / self.ZERO_PADDING
        mu    = mu_norm * (self.Z_MAX - self.Z_MIN) + self.Z_MIN
        ehat  = ehat_norm * self.MAX_VALUE
        phi   = torch.remainder(phi_norm, 1) * 2 * math.pi

        return sigma, mu, ehat, phi

    # -----------------------------------------------------------------------
    def scaling_unfold(self, sigma_norm, mu_norm, ehat_norm, phic_norm, phis_norm):
        # To scale up all normalized parameters
        sigma = sigma_norm / self.ZERO_PADDING
        mu    = mu_norm * (self.Z_MAX - self.Z_MIN) + self.Z_MIN
        ehat  = ehat_norm * self.MAX_VALUE
        phic = (phic_norm - 0.5) * 2.0
        phis = (phis_norm - 0.5) * 2.0
        
        return sigma, mu, ehat, phic, phis
    
    # -----------------------------------------------------------------------
    def logit(self, x):
#         eps = 1e-10
#         z = torch.clamp(x, min = eps, max = 1.0 - eps)
#         return torch.log(z) - torch.log1p(-z)
        return torch.logit(x, eps=1e-4)
    
    # -----------------------------------------------------------------------
    def projection(self, sigma_one, mu_one, ehat_one, phic_one, phis_one):
        sigma = torch.sigmoid(sigma_one)
        mu = torch.sigmoid(mu_one)
        ehat = torch.sigmoid(ehat_one)
        phic = torch.sigmoid(phic_one)
        phis = torch.sigmoid(phis_one)
        
        return sigma, mu, ehat, phic, phis

    # -----------------------------------------------------------------------
    def forward_channels(self, batch_shape, batch_predict):
        # this is the forward model, as a decoder
        return model_real, model_imag, reg
    
    # -----------------------------------------------------------------------
    def Lq_norm(self, x, p, dim, keepdim=False):
        return torch.pow(torch.mean(torch.pow(torch.abs(x) + self.EPS, p), dim=dim, keepdim=keepdim), 1.0/p)
    
    # -----------------------------------------------------------------------
    def L1_norm(self, x):
        return torch.mean(torch.abs(x))
    
    # -----------------------------------------------------------------------
    def p_zero(self):
        return 0.0
        
    # -----------------------------------------------------------------------
    def forward(self, batch_shape, batch_predict):
        # this is the forward model, as a decoder
        model_predict = torch.zeros(batch_shape, dtype=torch.float, requires_grad=True, device=self.device).clone()
        
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
        model_predict[:, self.INDEX_REAL, :, :, :] =      ehat * sinc_sigma_mu * torch.cos(omega_z_phi) 
        model_predict[:, self.INDEX_IMAG, :, :, :] = -1 * ehat * sinc_sigma_mu * torch.sin(omega_z_phi)
        
        with torch.no_grad():
            # if the parameter difference should be calculated this epoch
            self.sigma = sigma_one.clone().requires_grad_(False).squeeze_()
            self.mu = mu_one.clone().requires_grad_(False).squeeze_()
            self.ehat = ehat_one.clone().requires_grad_(False).squeeze_()
            self.phi = phi_one.clone().requires_grad_(False).squeeze_()
            self.kernel = None
            self.tv = None
            
        return model_predict
    
    # -----------------------------------------------------------------------
    def forward_perpixel(self, batch_shape, batch_predict):
        # batch_prediction format: [batch_size, channel, parameters, x, y]
        model_predict = torch.zeros(batch_shape, dtype=torch.float, requires_grad=True, device=self.device).clone()

        """
        Model based decoder
        """
        sigma_norm = batch_predict[:, 0]
        mu_norm = batch_predict[:, 1]
        ehat_norm = batch_predict[:, 2]
        phi_norm = batch_predict[:, 3]

        # calculate all predicted parameters
        sigma, mu, ehat, phi = self.scaling(sigma_norm, mu_norm, ehat_norm, phi_norm)

        # extend all parameters in z-axis
        sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)
        mu = mu.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)
        ehat = ehat.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)
        phi = phi.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)

        # calculate sinc(x) = sin(pi * x) / (pi * x)
        pi_sigma_mu = math.pi * sigma * (self.z_tensor - mu)
        sin_pi_sigma_mu = torch.sin( math.pi * sigma * (self.z_tensor - mu) )
        
        pi_sigma_mu[ pi_sigma_mu == 0 ] = 1
        sin_pi_sigma_mu[ pi_sigma_mu == 0 ] = 1
        
        sinc_sigma_mu = (sin_pi_sigma_mu) / (pi_sigma_mu)

        # calculate phase item
        omega_z_phi = self.OMEGA * self.z_tensor - phi

        # compute the model
        model_real =      torch.abs(ehat) * sinc_sigma_mu * torch.cos(omega_z_phi) 
        model_imag = -1 * torch.abs(ehat) * sinc_sigma_mu * torch.sin(omega_z_phi)

        """
        Network kernel convolution
        """
        # optimize only for encoder
        model_predict[:, self.INDEX_REAL, :, :, :] = model_real.squeeze(1)
        model_predict[:, self.INDEX_IMAG, :, :, :] = model_imag.squeeze(1)

        return model_predict
    
    # -----------------------------------------------------------------------
    def forward_perpixel_unfold(self, batch_shape, batch_predict):
        # batch_prediction format: [batch_size, channel, parameters, x, y]
        model_predict = torch.zeros(batch_shape, dtype=torch.float, requires_grad=True, device=self.device).clone()

        """
        Model based decoder
        """
        sigma_norm = batch_predict[:, 0]
        mu_norm = batch_predict[:, 1]
        ehat_norm = batch_predict[:, 2]
        phic_norm = batch_predict[:, 3]
        phis_norm = batch_predict[:, 4]

        # calculate all predicted parameters
        sigma, mu, ehat, phic, phis = self.scaling_unfold(sigma_norm, mu_norm, ehat_norm, phic_norm, phis_norm)
        
        phi = torch.atan2( phis, phic )

        # extend all parameters in z-axis
        sigma = sigma.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)
        mu = mu.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)
        ehat = ehat.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)
        phi = phi.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, self.NZ, 1, 1)

        # calculate sinc(x) = sin(pi * x) / (pi * x)
        pi_sigma_mu = math.pi * sigma * (self.z_tensor - mu)
        sin_pi_sigma_mu = torch.sin( math.pi * sigma * (self.z_tensor - mu) )
        sin_pi_sigma_mu[ torch.abs(pi_sigma_mu) < self.FLT_EPSILON ] = 1
        pi_sigma_mu[ torch.abs(pi_sigma_mu) < self.FLT_EPSILON ] = 1
        sinc_sigma_mu = (sin_pi_sigma_mu) / (pi_sigma_mu)

        # calculate phase item
        omega_z_phi = self.OMEGA * self.z_tensor - phi

        # compute the model
        model_real =      torch.abs(ehat) * sinc_sigma_mu * torch.cos(omega_z_phi) 
        model_imag = -1 * torch.abs(ehat) * sinc_sigma_mu * torch.sin(omega_z_phi)

        """
        Network kernel convolution
        """
        # optimize only for encoder
        model_predict[:, self.INDEX_REAL, :, :, :] = model_real.squeeze(1)
        model_predict[:, self.INDEX_IMAG, :, :, :] = model_imag.squeeze(1)

        return model_predict
    
    # -----------------------------------------------------------------------
    def forward_with_init(self, batch_shape, batch_predict, N_init, u_init):
        # this is the forward model, as a decoder
        model_predict = torch.zeros(batch_shape, dtype=torch.float, requires_grad=True, device=self.device).clone()
        
        """
        Model based decoder
        """       
        batch_predict_with_init = (batch_predict - N_init) * 0.10 + u_init
        sigma_norm = batch_predict_with_init[:, :, 0, :, :]
        mu_norm = batch_predict_with_init[:, :, 1, :, :]
        ehat_norm = batch_predict_with_init[:, :, 2, :, :]
        phic_norm = batch_predict_with_init[:, :, 3, :, :]
        phis_norm = batch_predict_with_init[:, :, 4, :, :]

        # do projection
        if (self.SHOULD_PROJECT_BEFORE):
            sigma_proj, mu_proj, ehat_proj, phic_proj, phis_proj = self.projection(sigma_norm, mu_norm, ehat_norm, phic_norm, phis_norm)
        else:
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
        model_predict[:, self.INDEX_REAL, :, :, :] =      ehat * sinc_sigma_mu * torch.cos(omega_z_phi) 
        model_predict[:, self.INDEX_IMAG, :, :, :] = -1 * ehat * sinc_sigma_mu * torch.sin(omega_z_phi)
        
        with torch.no_grad():
            # if the parameter difference should be calculated this epoch
            self.sigma = sigma_one.clone().requires_grad_(False).squeeze_()
            self.mu = mu_one.clone().requires_grad_(False).squeeze_()
            self.ehat = ehat_one.clone().requires_grad_(False).squeeze_()
            self.phi = phi_one.clone().requires_grad_(False).squeeze_()
            self.kernel = None
            self.tv = None
            
        return model_predict
    
    # -----------------------------------------------------------------------
    def initialize_u(self, batch, use_logit = True):
        z_reshape = self.z_np.reshape((1, 1, self.NZ, 1, 1))
        z_tile = np.tile(z_reshape.squeeze(0).squeeze(0), (1, self.NX, self.NY))
        
        sigma_norm = torch.zeros([1, 1, self.NX, self.NY], dtype=torch.float)
        mu_norm = torch.zeros([1, 1, self.NX, self.NY], dtype=torch.float)
        ehat_norm = torch.zeros([1, 1, self.NX, self.NY], dtype=torch.float)
        phic_norm = torch.zeros([1, 1, self.NX, self.NY], dtype=torch.float)
        phis_norm = torch.zeros([1, 1, self.NX, self.NY], dtype=torch.float)
        
        data_real = batch[0, self.INDEX_REAL, :, :, :].cpu().detach().numpy()
        data_imag = batch[0, self.INDEX_IMAG, :, :, :].cpu().detach().numpy()
        data_complex = data_real + 1j * data_imag
        
        sin_component = np.sum( np.sin(self.OMEGA * z_tile + np.angle(data_complex)), axis=0)
        cos_component = np.sum( np.cos(self.OMEGA * z_tile + np.angle(data_complex)), axis=0)
        
        phi_init = np.arctan2( sin_component, cos_component )
        phic_init = np.cos(phi_init)
        phis_init = np.sin(phi_init)

        mag = torch.sqrt(torch.pow(batch[0, self.INDEX_REAL, :, :, :], 2) + torch.pow(batch[0, self.INDEX_IMAG, :, :, :], 2))
        magmax, magmax_i = torch.max(mag, dim=0)

        sigma_norm[0, 0, :, :] = 1.0
        mu_norm[0, 0, :, :] = magmax_i / float(self.NZ)
        ehat_norm[0, 0, :, :] = magmax / self.MAX_VALUE
        phic_norm[0, 0, :, :] = torch.from_numpy(phic_init)
        phis_norm[0, 0, :, :] = torch.from_numpy(phis_init)
        
        # store to a single tensor
        u_tensor = torch.zeros((1, 1, self.NP, self.NX, self.NY), dtype=torch.float, requires_grad = False)
        u_tensor[:, :, 0, :, :] = sigma_norm[0, 0, :, :]
        u_tensor[:, :, 1, :, :] =    mu_norm[0, 0, :, :]
        u_tensor[:, :, 2, :, :] =  ehat_norm[0, 0, :, :]
        u_tensor[:, :, 3, :, :] =  phic_norm[0, 0, :, :]
        u_tensor[:, :, 4, :, :] =  phis_norm[0, 0, :, :]

        if (use_logit):
            return self.logit(u_tensor)
        else:
            return u_tensor
    
    # -----------------------------------------------------------------------
    def initialize_u_random(self, use_logit = False):
        # store to a single tensor
        u_tensor = torch.rand((1, 1, self.NP, self.NX, self.NY), dtype=torch.float, requires_grad = False)

        if (use_logit):
            return self.logit(u_tensor)
        else:
            return u_tensor
    
    # -----------------------------------------------------------------------
    def initialize_u_zeros(self):
        # store to a single tensor
        u_tensor = torch.zeros((1, 1, self.NP, self.NX, self.NY), dtype=torch.float, requires_grad = False)

        return u_tensor
    
    # -----------------------------------------------------------------------
    def get_pixel(self, pixel_x, pixel_y, data_tensor, model_tensor):
        axis_base = self.z_np
        legend_left = 'real'
        data_left = data_tensor[:, self.INDEX_REAL, :, pixel_x, pixel_y].squeeze_().cpu().detach()
        model_left = model_tensor[:, self.INDEX_REAL, :, pixel_x, pixel_y].squeeze_().cpu().detach()
        
        legend_right = 'imag'
        data_right = data_tensor[:, self.INDEX_IMAG, :, pixel_x, pixel_y].squeeze_().cpu().detach()
        model_right = model_tensor[:, self.INDEX_IMAG, :, pixel_x, pixel_y].squeeze_().cpu().detach()
        
        return axis_base, data_left, model_left, data_right, model_right, legend_left, legend_right
    
    # -----------------------------------------------------------------------
    def save_mat(self, path_file, sigma = None, mu = None, ehat = None, phi = None, loss_map = None):
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
            
        matfiledata = {} # make a dictionary to store the MAT data in
        matfiledata[u'Ehat'] = ehat.cpu().detach().numpy()
        matfiledata[u'Sigma'] = sigma.cpu().detach().numpy()
        matfiledata[u'Mu'] = mu.cpu().detach().numpy()
        matfiledata[u'Phi'] = phi.cpu().detach().numpy()
            
        if (loss_map is not None):
            matfiledata[u'Loss'] = loss_map.cpu().detach().numpy()

        scipy.io.savemat(path_file, matfiledata)

        if (self.verbose):
            print("{} is saved".format(path_file))
            
        return path_file