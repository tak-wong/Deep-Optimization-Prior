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

def util_write_log_header(DIR_LOGS, STR_TIMESTP, verbose = False):
    PATH_LOG = os.path.join(DIR_LOGS, STR_TIMESTP + ".txt")
    #if os.path.isfile(PATH_LOG):
    #    return PATH_LOG
    fw = open(PATH_LOG, 'a+')
    content = "{:^23}|{:^24}|{:^11}|{:^11}|{:^12}|{:^25}| {:<}".format(
        "start_time", "end_time", "epoch_num", "batch_num", "eval_type", "loss", "remark")
    fw.write(content + "\n")
    fw.close()
    if verbose:
        print(" ".join(content.split()))
        
    return PATH_LOG

def util_write_log(PATH_LOG, start_time, end_time, epoch_num, batch_num, eval_type, loss, remark="", verbose = False):
    START_TIME_STR = start_time.strftime("%Y%m%d_%H%M%S_%f")
    END_TIME_STR = end_time.strftime("%Y%m%d_%H%M%S_%f")
    
    # eval_type = batch, training, validation
    
    fw = open(PATH_LOG, 'a+')
    content = "{:^23}|{:^24}|{:^11}|{:^11}|{:^12}|{:^25.16f}| {:<}".format(
        START_TIME_STR, END_TIME_STR, epoch_num, batch_num, eval_type, loss, remark)
    fw.write(content + "\n")
    fw.close()
    if verbose:
        print(" ".join(content.split()))
        
    return PATH_LOG
