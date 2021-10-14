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

from .dataset_thz import *

class dataset_thz_perpixel(dataset_thz):
    # -----------------------------------------------------------------------
    def __init__(self, path_data, device, verbose = True):
        super(dataset_thz_perpixel, self).__init__(path_data, device, verbose)
        
        # start of initialization
        self.load_data_perpixel()

        if (self.verbose):
            print('dataset_thz_perpixel.init')
        
    # -----------------------------------------------------------------------
    def load_data_perpixel(self):
        # load the data from file
        """
        Now we divide them into sets in z-direction
        """
        self.data_size = self.NX * self.NY # number of data
        self.data_tensor = torch.zeros((self.NC, self.NZ, self.data_size, 1, 1), dtype=torch.float).to(self.device)
        
        for x in range(self.NX):
            for y in range(self.NY):
                data = self.data_np_complex[x, y, :]

                # load the matrix in (C,X,Y,Z)
                data_numpy = np.ndarray(shape=(self.NC, self.NZ), dtype=float)
                data_numpy[self.INDEX_REAL, :] = data['real']
                data_numpy[self.INDEX_IMAG, :] = data['imag']

                # convert to (C,Z,X,Y)
                data_index = x * self.NX + y

                self.data_tensor[:, :, data_index, 0, 0] = torch.from_numpy(data_numpy).float().cuda()

        if self.verbose:
            print("data_tensor shape: {}".format(self.data_tensor.shape))

    # -----------------------------------------------------------------------
    def __len__(self):
        return self.data_size
    
    # -----------------------------------------------------------------------
    def __getitem__(self, index):
        return self.data_tensor[:, :, index, :, :]

    # -----------------------------------------------------------------------
    def get_method(self):
        return "per_pixel"
    
    # -----------------------------------------------------------------------
    def get_NX(self):
        return self.NX
    
    # -----------------------------------------------------------------------
    def get_NY(self):
        return self.NY
    
    # -----------------------------------------------------------------------
    def get_shape(self):
        return (self.NC, self.NZ, 1, 1)
    