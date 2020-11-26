#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:54:34 2019

@author: Michele Svanera
University of Glasgow.

"""

################################################################################################################
## Imports 

from __future__ import division, print_function

import logging
log = logging.getLogger(__name__)

from time import strftime, localtime
import os, sys
import nvidia_smi

import numpy as np
import json
import nibabel as nib
import pandas as pd

import keras
import tensorflow as tf


################################################################################################################
## Paths and Constants

# Remove warnings on font
logging.getLogger('matplotlib.font_manager').disabled = True


################################################################################################################
## Functions

    
def initialiseLogger(log_filename, with_time=False, log_level=logging.NOTSET):
    '''
    Initialise log file.
    IN: 
        log_filename: full_path
    OUT:
        logger
    '''
    
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    local_time_str = strftime("%Y-%m-%d %H:%M:%S", localtime())
    log_filename_with_time = (local_time_str.replace(':','-')).replace(' ','_')

    # Logger setup
    if with_time:
        log_filename = log_filename[:-4] + '_' + log_filename_with_time + '.log'
    log_filename = checkFileOut(log_filename)
        
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)8s: %(message)s","%Y/%m/%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("------------- " + log_filename_with_time + " -------------\n")
    
    return logger


def checkFileOut(filename):
    """
    Function which checks that a file exists or add a '_N' at the end.
    IN:
        * filename (with extension)
        * filename_N (with extension)
    """
    
    filename_out = filename 

    n_file_name = 0    
    while(os.path.exists(filename_out) is True):
        n_file_name += 1
        filename_out = filename[:-4] + '_' + str(n_file_name) + filename[-4:]
        
    return filename_out


# Starting from Out_dir, creates any folder and subfolders in path_out_fig
def createFoldersIteratevely(Out_dir, path_out_fig):

    x = ''
    for p_path in path_out_fig.split('/')[:-1]:
        x += (p_path+'/')
        
        try:
            os.mkdir(Out_dir + x)
        except:
            print('',end='')

    return



def checkAndCreateFolderOut(filename):
    """
    Function which checks that a folder exists or add a '_N' at the end.
    IN:
        * filename        
    OUT:
        * filename
    """
    
    filename_out = filename 

    n_file_name = 0    
    while(os.path.exists(filename_out) is True):
        n_file_name += 1

        if filename[-1] == '/':
            filename_out = filename[:-1] + '_' + str(n_file_name)
        else:
            filename_out = filename + '_' + str(n_file_name)
    
    os.mkdir(filename_out)
    
    if filename_out[-1] == '/':
        return filename_out
    else:
        return filename_out + '/'
    
    
def initialiseOutputFolder(out_folder,log_description='',initial_desciption='training_'):
    
    # Compose output name
    local_time_str = strftime("%Y-%m-%d %H:%M:%S", localtime())
    log_filename_with_time = (local_time_str.replace(':','-')).replace(' ','_')
    out_folder_complete = initial_desciption + log_filename_with_time + '_' + log_description
    out_folder_complete = out_folder_complete.replace('/','-')
    
    out_folder_complete = checkAndCreateFolderOut(out_folder + out_folder_complete)

    return out_folder_complete


def saveCsvWithDatasetList(list_of_fullpath_names, y_label, namefile):
    '''
    Utility to save a CSV file with a list of filenames.
    '''
    dataset_name = (os.path.basename(namefile)).split('_')[0]
    df = pd.DataFrame({dataset_name: list_of_fullpath_names,
                       'GT': y_label})

    df.to_csv(namefile, index=False)
    
    return


def loadCsvWithDatasetList(namefile):
    '''
    Utility to load a CSV file with a list of filenames.
    '''
    
    dataset = pd.read_csv(namefile)
    
    dataset_name = (os.path.basename(namefile)).split('_')[0]
    X_fullnames = list(dataset[dataset_name])
    y = list(dataset['GT'])
    
    return X_fullnames, y


def logEveryPackageLoad(log):
    log.info('%10s : %s' % ('Running on', os.uname()[1]))
    log.info('%10s : %s' % ('Python', sys.version.split('\n')[0]))
    log.info('%10s : %s' % ('Numpy', np.__version__))
    log.info('%10s : %s' % ('json', json.__version__))
    log.info('%10s : %s' % ('nibabel', nib.__version__))
    log.info('%10s : %s' % ('Keras', keras.__version__))
    log.info('%10s : %s \n' % ('Tensorflow', tf.__version__))


def checkGPUsAvailability(n_gpus=1):
    '''
    Test that GPUs have free memory on 'n_gpus'.
    OUT:
        True: if they have
        False: if not
    '''
    # For every gpu to check
    for i_gpu in range(n_gpus):
        
        # Access to the memory used by the i-th gpu
        try:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i_gpu)
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            print('Warning: GPU did not accessed')
            break
                
        # If more than 1GB is taken, then stop
        if (mem_res.used/(1024.**3) > 1.0):         # greater than 1GB of VRAM
            # Report it
            print('Memory used (gpu-%i): %.2f GB' % (i_gpu, mem_res.used/(1024**3)), end='')
            print(' - on total: %.2f GB' % (mem_res.total/(1024**3)))
            return False

    return True
















