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
import shutil
import os

def util_copy_scripts(DIR_OUTPUT, DIR_SCRIPTS, verbose=False):
    # copy all sub-directories
    directories = []
    for entry in os.scandir('.'):
        if entry.is_dir() and not entry.name.startswith("_") and not entry.name.startswith("."):
            directories.append(entry.name)

    for directory in directories:
        src_path = os.path.join('.', directory)
        dst_path = os.path.join(DIR_SCRIPTS, directory)
        dst = shutil.copytree(src_path, dst_path)
        if verbose:
            print("copy to " + dst)
        
    # add all python files into script directory
    files_py = []
    for entry in os.scandir('.'):
        if entry.is_file() and entry.name.endswith(".py"):
            files_py.append(os.path.join('.', entry.name))
    
    for file in files_py:
        dst = shutil.copy2(file, DIR_SCRIPTS)
        if verbose:
            print("copy to " + dst)
        
    # add all matlab files into output directory
    files_m = []
    for entry in os.scandir('.'):
        if entry.is_file() and entry.name.endswith(".m") and not entry.name.startswith("_"):
            files_m.append(os.path.join('.', entry.name))
    
    for file in files_m:
        dst = shutil.copy2(file, DIR_OUTPUT)
        if verbose:
            print("copy to " + dst)
            
    # add all python notebook files into output directory
    files_ipynb = []                   
    for entry in os.scandir('.'):
        if entry.is_file() and entry.name.endswith(".ipynb") and not entry.name.endswith("-checkpoint.ipynb") and not entry.name.startswith("_"):
            files_ipynb.append(os.path.join('.', entry.name))
    
    for file in files_ipynb:
        dst = shutil.copy2(file, DIR_OUTPUT)
        if verbose:
            print("copy to " + dst)
    