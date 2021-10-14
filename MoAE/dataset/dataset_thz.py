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
import h5py
import numpy as np
import torch

from .dataset import *

class dataset_thz(dataset):
    # -----------------------------------------------------------------------
    def __init__(self, path_data, device, verbose = True):
        super(dataset_thz, self).__init__(path_data, device, verbose)
        
        # start of initialization
        self.load_data()

        if self.verbose:
            print('dataset_thz.init')
            
    # -----------------------------------------------------------------------
    def load_data(self):
        # load the data from file
        with h5py.File(self.path_data, 'r') as hf:
            self.data_np_complex = hf['data'][:]
            self.z_np = np.array(hf['vector_z'][0]).astype(np.float)
            self.NC = np.array(hf['NC'][0]).astype(np.int)[0]
            self.NZ = np.array(hf['NZ'][0]).astype(np.int)[0]
            self.NX = np.array(hf['NX'][0]).astype(np.int)[0]
            self.NY = np.array(hf['NY'][0]).astype(np.int)[0]
            self.INDEX_REAL = np.array(hf['INDEX_REAL'][0]).astype(np.int)[0]
            self.INDEX_IMAG = np.array(hf['INDEX_IMAG'][0]).astype(np.int)[0]
            self.OMEGA = np.array(hf['OMEGA'][0]).astype(np.float)[0]
            self.ZERO_PADDING = np.array(hf['ZERO_PADDING'][0]).astype(np.float)[0]
            self.MAX_VALUE = np.array(hf['MAX_VALUE'][0]).astype(np.float)[0]
            hf.close()
            
        self.data_np = np.ndarray(shape=(self.NC, self.NY, self.NX, self.NZ), dtype=float)
        self.data_np[self.INDEX_REAL, :, :, :] = self.data_np_complex['real']
        self.data_np[self.INDEX_IMAG, :, :, :] = self.data_np_complex['imag']
            
        # load the data from numpy to pytorch tensor
        self.data_tensor = torch.from_numpy(self.data_np).permute(0, 3, 2, 1).float().to(self.device).unsqueeze(0)
        
        if self.verbose:
            print("data_tensor shape: {}".format(self.data_tensor.shape))

    # -----------------------------------------------------------------------
    def __len__(self):
        return 1
    
    # -----------------------------------------------------------------------
    def __getitem__(self, index):
        return self.data_tensor
    
    # -----------------------------------------------------------------------
    def get_method(self):
        return "complex_matrix"

    # -----------------------------------------------------------------------
    def get_shape(self):
        return (self.NC, self.NZ, self.NX, self.NY)
    