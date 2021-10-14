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
import datetime
import os

def util_make_folders(path_home, dir_setting, verbose = False):
    PATH_OUTPUT = os.path.join(path_home, dir_setting)
    
    # time stamp string
    STR_TIMESTP = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    
    DIR_OUTPUT = os.path.join(PATH_OUTPUT, STR_TIMESTP)
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)
        if verbose:
            print("make folder at " + DIR_OUTPUT)

    DIR_SCRIPTS = os.path.join(DIR_OUTPUT, "scripts")
    if not os.path.exists(DIR_SCRIPTS):
        os.makedirs(DIR_SCRIPTS)
        if verbose:
            print("make folder at " + DIR_SCRIPTS)

    DIR_CHECKPOINTS = os.path.join(DIR_OUTPUT, "checkpoints")
    if not os.path.exists(DIR_CHECKPOINTS):
        os.makedirs(DIR_CHECKPOINTS)
        if verbose:
            print("make folder at " + DIR_CHECKPOINTS)

    DIR_LOGS = os.path.join(DIR_OUTPUT, "logs")
    if not os.path.exists(DIR_LOGS):
        os.makedirs(DIR_LOGS)
        if verbose:
            print("make folder at " + DIR_LOGS)
        
    return STR_TIMESTP, DIR_OUTPUT, DIR_SCRIPTS, DIR_CHECKPOINTS, DIR_LOGS

def util_load_folders(path_home, verbose=False):
    
    DIR_CHECKPOINTS = os.path.join(path_home, "checkpoints")
    if verbose:
        print("load folder at " + DIR_CHECKPOINTS)
    
    DIR_LOGS = os.path.join(path_home, "logs")
    if verbose:
        print("load folder at " + DIR_LOGS)
        
    # find the latest run, that was interrupted
    RUN = 0
    for entry in os.scandir(DIR_LOGS):
        n = int(entry.path[-2:])
        RUN = n if n > RUN else RUN
    if verbose:
        print("found most recent run: {}".format(RUN))
        
    # find the timestamp for the latest run
    for file in os.listdir( os.path.join(DIR_LOGS, "run_{:02}".format(RUN)) ):
        if file.endswith(".txt"):
            STR_TIMESTP = file.split(".")[0]
    if verbose:
        print("load timestamp: " + STR_TIMESTP)
    
    return STR_TIMESTP, DIR_CHECKPOINTS, DIR_LOGS, RUN