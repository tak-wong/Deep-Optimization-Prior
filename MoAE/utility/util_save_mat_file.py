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
import h5py
import scipy.io
import numpy

def util_save_mat_file(matrix, matrix_name, filename):
    matfiledata = {} # make a dictionary to store the MAT data in
    matfiledata[matrix_name] = matrix
    
    scipy.io.savemat(filename, matfiledata)
    
def util_save_mat_file_all(matfiledata, filename):
    scipy.io.savemat(filename, matfiledata)