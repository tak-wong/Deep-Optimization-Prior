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
import os
import datetime

def util_write_summary_header(DIR_OUTPUT, STR_TIMESTP, str_dataset_name, str_dataset_method, str_model, str_network, str_lr, verbose = False):
    # eval_type = batch, training, validation
    PATH_SUMMARY = os.path.join(DIR_OUTPUT, 'summary_{}.txt'.format(STR_TIMESTP))
    
    fw = open(PATH_SUMMARY, 'a+')
    content = "timestp: {}\n".format(STR_TIMESTP)
    content += "dataset_name: {}\n".format(str_dataset_name)
    content += "dataset_method: {}\n".format(str_dataset_method)
    content += "model: {}\n".format(str_model)
    content += "network: {}\n".format(str_network)
    content += "learning rate: {}\n".format(str_lr)
    content += "-------------------\n"
    content += "{},{},{},{},\n".format("run", "loss", "runtime", "remark")
    fw.write(content)
    fw.close()
    if verbose:
        print(" ".join(content.split()))
        
    return PATH_SUMMARY
        
def util_write_summary(PATH_SUMMARY, run, loss, runtime, remark = "", verbose = False):
    # eval_type = batch, training, validation
    
    fw = open(PATH_SUMMARY, 'a+')
    content = "{},{:.16f},{:.8f},{},\n".format(run, loss, runtime, remark)
    fw.write(content)
    fw.close()
    if verbose:
        print(" ".join(content.split()))
        
    return PATH_SUMMARY

def util_write_network_summary(DIR_OUTPUT, STR_TIMESTP, str_network, str_network_summary, verbose = False):
    # eval_type = batch, training, validation
    PATH_SUMMARY = os.path.join(DIR_OUTPUT, 'summary_network_{}.txt'.format(STR_TIMESTP))
    
    fw = open(PATH_SUMMARY, 'w+')
    content = "timestp: {}\n".format(STR_TIMESTP)
    content += "network: {}\n".format(str_network)
    content += "-------------------\n"
    content += "network summary:\n"
    content += "{}\n".format(str_network_summary)
    fw.write(content)
    fw.close()
    if verbose:
        print(" ".join(content.split()))
        
    return PATH_SUMMARY