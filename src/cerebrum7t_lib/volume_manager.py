#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:51:53 2019

@author: Michele Svanera
University of Glasgow.

Functions to load and manage Volume data.
"""

################################################################################################################
## Imports 

from __future__ import division, print_function

import logging
log = logging.getLogger(__name__)

import os, sys
from os.path import join as opj

import numpy as np
import nibabel as nib
import pandas as pd

from keras.utils import np_utils

################################################################################################################
## Functions

def findListOfAnatomical(path_in, identifier='nii.gz'):

    all_anat = []
    for root, _, files in os.walk(path_in):
        for i_file in files:
            if i_file.endswith(identifier):
                all_anat.append(root + '/' + i_file)
    
    return sorted(list(np.unique(all_anat)))


def findCorrespondingGTPath(full_path_GT, full_path_in_vol, all_volume_fullpathname, GT_to_predict):
    
    all_anat_GT = []
    for i_vol in all_volume_fullpathname:
        
        # Find the correspondent ID 
        subj_id, _, anat_name = i_vol.replace(full_path_in_vol,'').split('/')
        all_anat_GT.append(full_path_GT + subj_id + '/' + GT_to_predict)

    return all_anat_GT
    

def findCorrespondingGTPathBIDS(all_volume_fullpathname, GT_to_predict):
    
    all_anat_GT = []
    for i_vol in all_volume_fullpathname:
        
        # Find the correspondent ID 
        subj_id, sess_id = os.path.basename(i_vol).split('_')[:2]        # ex. 'sub-001_ses-001_T1w.nii.gz'

        # Compose GT
        gt_filename = '_'.join([subj_id, sess_id, GT_to_predict])               # ex. 'sub-001_ses-001_Fracasso16-mc.nii.gz'
        gt_fullname = opj(os.path.dirname(i_vol).replace('anat', 'seg'), gt_filename)
        all_anat_GT.append(gt_fullname)

    return all_anat_GT


def excludeTestingVolume(all_volumes, testing_volume, full_path_in_vol):
    ''' Given a list of all volumes, exclude the testing volume.
        Used for behavioural testing volumes.
    '''    
    
    out_list = []
    excluded_list = []
    for i_vol in all_volumes:
        i_vol_ID = (i_vol.split(full_path_in_vol)[1]).split('/')[0]
        if not i_vol_ID in testing_volume:
            out_list.append(i_vol)
        else:
            excluded_list.append(i_vol)

    return out_list, excluded_list
    

def excludeTestingVolumeBIDS(all_volumes, testing_volume):
    ''' Given a list of all volumes, exclude the testing volume.
        Used for behavioural testing volumes.
    '''    
    
    out_list = []
    excluded_list = []
    for i_vol in all_volumes:
        
        if np.sum([1 for j_test in testing_volume if j_test in i_vol]) == 0:
            out_list.append(i_vol)
        else:
            excluded_list.append(i_vol)

    return out_list, excluded_list


def volumeZscoring(input_vol, voxelwise_mean, voxelwise_std):
    '''
    Function to z-scoring of loaded-by-generator data 
    '''
    
    # Standardize each training volume
    input_vol -= voxelwise_mean
    
    # Prevent division by zero
    voxelwise_std[voxelwise_std == 0] = 1
    input_vol /= voxelwise_std
    
    return input_vol


def checkVolumeIntegrity(input_vol, vol_name):
    '''
    Function to check volume integrity
    '''
    
    n_nan = np.sum(np.isnan(input_vol))
    n_inf = np.sum(np.isinf(input_vol))
    
    if n_nan != 0:
        log.info('ERROR 23.1: %d NaN(s) found in volume "%s"!' % (n_nan, vol_name))
        sys.exit(0)
        
    if n_inf != 0:
        log.info('ERROR 23.1: %d inf(s) found in volume "%s"!' % (n_inf, vol_name))
        sys.exit(0)
        

def checkVolumeGTnameCorrespondence(vol_name, GT_name):
    '''
    Function to check that the GT and the volume match based on the ID.
    '''

    # works with both: original volume and augmented
    root_path = findCommonPath(vol_name, GT_name)
    anat_folder = (vol_name.split(root_path)[1]).split('/')[0]
    gt_folder = (GT_name.split(root_path)[1]).split('/')[0]
    
    vol_id = (vol_name.split(anat_folder+'/')[1]).split('/')[0]
    gt_id = (GT_name.split(gt_folder+'/')[1]).split('/')[0]
    
    if vol_id[-7:] == '.nii.gz':    # if there is still the extension
        vol_id = vol_id[:-7]
    
    if vol_id != gt_id:
        log.info('ERROR 24.1: Volume: "%s" and GT: "%s" do NOT match!' % (vol_id, gt_id))
        sys.exit(0)
        

def findCommonPath(str1, str2):
    ''' Takes 2 strings (paths) and find the root that they have in common
    '''
    if len(str1) < len(str2):
        min_str = str1
        max_str = str2
    else:
        min_str = str2
        max_str = str1
    
    matches = []
    min_len = len(min_str)
    for b in range(min_len):
        for e in range(min_len, b, -1):
            chunk = min_str[b:e]
            if chunk in max_str:
                matches.append(chunk)

    return os.path.dirname(max(matches, key=len))
           

def loadSingleVolume(vol_filename, voxelwise_mean, voxelwise_std, data_dims):
    '''
    Function to load and check a single volume.
    IN: 
        vol_filename: full path name of the volume to load
    OUT:
        vol: loaded volume 
    '''

    # load the actual volume
    vol = np.array(nib.load(vol_filename).get_data()).astype(dtype='float32')

    # check if everything is ok
    if vol.shape != data_dims:
        log.info('ERROR 23.2.1: volume "%s" size mismatch: exiting...'%(vol_filename))
        sys.exit(0)

    vol = volumeZscoring(vol, voxelwise_mean, voxelwise_std)

    checkVolumeIntegrity(vol, vol_filename)

    return vol


def loadSingleGTVolume(vol_filename, data_dims):
    '''
    Function to load and check a single volume.
    IN: 
        vol_filename: full path name of the volume to load
    OUT:
        vol: loaded volume 
    '''

    # load the actual volume (1 single class to predict)
    vol = np.array(nib.load(vol_filename).get_data()).astype(dtype='uint8')
    
    # if vol is a probability vol, then error (to difficult to deal with it here)
    if len(vol.shape) > 3:  
        log.info('ERROR 23.2.2: volume "%s" has more than 3 dimensions: exiting...'%(vol_filename))
        sys.exit(0)

    # check if everything is ok
    if vol.shape != data_dims:
        log.info('ERROR 23.2.3: volume "%s" size mismatch: exiting...'%(vol_filename))
        sys.exit(0)

    checkVolumeIntegrity(vol, vol_filename)

    return vol


def loadDatasetFromListFiles(full_path_volumes, y, voxelwise_mean, voxelwise_std, data_dims, num_segmask_labels=0):
    '''
    Function to load a list of volumes with related GT.
    IN: 
        full_path_volumes: full path name of the volumes to load
        voxelwise_mean, voxelwise_std: matrices OR list with multiple mean and std from different sites
        y: full path name of the GT to load
    OUT:
        X_data: loaded volumes (tf format)
        Y_data: loaded GT (tf format)
    '''
        
    def to_categorical_tensor(x3d, n_cls):
        x, y, z = x3d.shape
        x1d = x3d.ravel()
        y1d = np_utils.to_categorical(x1d, num_classes=n_cls)
        y4d = y1d.reshape([x, y, z, n_cls])
        return y4d

    X_data = []
    y_data = []
    
    # Load the first to find the N_classes (background included) -> WARNING: what if a class is not present!!
    if num_segmask_labels == 0:         # if not specified, compute it dinamically
        num_segmask_labels = np.unique(loadSingleGTVolume(y[0], data_dims)).shape[0]
    
    for vol_fullFilename, GT_fullFilename in zip(full_path_volumes,y):
        
        # Take care of the case in which 
        if type(voxelwise_mean) == dict:
            # Discover which mean and std use based on the name of the db
            i_db = (vol_fullFilename.split('T1/')[1]).split("_")[0] 
            
            i_voxelwise_mean = voxelwise_mean[i_db]
            i_voxelwise_std = voxelwise_std[i_db]
        else:
            i_voxelwise_mean = voxelwise_mean
            i_voxelwise_std = voxelwise_std

        # Load the volume with relative GT
        i_vol_loaded = loadSingleVolume(vol_fullFilename, i_voxelwise_mean, i_voxelwise_std, data_dims)
        i_GT_loaded = loadSingleGTVolume(GT_fullFilename, data_dims)
        
        # Prepare data for the network
        i_vol_loaded = i_vol_loaded.reshape((data_dims[0], data_dims[1], data_dims[2], 1))
        i_GT_loaded = to_categorical_tensor(i_GT_loaded, num_segmask_labels).astype(dtype='uint8')#'int32')
        
        # Populate the list
        X_data.append(i_vol_loaded)
        y_data.append(i_GT_loaded)
        
        # Check new classes didn't show up
        checkVolumeGTnameCorrespondence(vol_fullFilename, GT_fullFilename)
        
    # Concatenate data
    X_data = np.stack(X_data,axis=0)
    Y_data = np.stack(y_data,axis=0)

#    np.save('X_data.npy',X_data)
#    np.save('Y_data.npy',Y_data)
    
    return X_data, Y_data


def fromGTtoYlabels(GT, value_to_classify, vol_names):
    '''
    Function to extract the 'y' from the 'GT' file.
    It matches the 'session' in the 'GT'.
    IN:
        GT: pandas file with the GT
    OUT:
        vol_names_with_GT: list of volumes of which we have the GT required
        y: GT
    '''

    # Create a pandas file starting from 'vol_names'
    filename_of_volumes = [os.path.basename(i_fullpath) for i_fullpath in vol_names]   
    df_subject_names = pd.DataFrame({'filename': filename_of_volumes})
    
    # Try to match the 'df_subject_names' with the 'GT' loaded
    df = pd.merge(GT,
             df_subject_names,
             how='inner',
             on=['filename'])
    
    # Parse output
    y = list(df[value_to_classify])
    
    vol_names_with_GT = list(df['filename'])
    vol_names_with_GT = [os.path.dirname(vol_names[0])+'/'+i for i in vol_names_with_GT]
    
    return vol_names_with_GT, y
    

def convertGTtoNumerical(y, GT_to_predict):
    '''
    Convert string GT (ex. 'M' or 'F') to numerical (ex. 0, 1)
    
    '''
    
    unique_values = np.unique(y)
    n_classes_classification = unique_values.shape[0]
    
    y_out = np.zeros((len(y),), dtype=int)
    dictionary_GT = {}
    
    for i_indx, i_value in enumerate(unique_values):

        dictionary_GT[i_value] = i_indx
        y_out[[i == i_value for i in y]] = i_indx

    return y_out, n_classes_classification, dictionary_GT


def checkNanValues(all_fullnames, y):
    ''' Check if there are NaN values in the GT and delete the row.
    IN:
        all_fullnames
        y
    OUT:
        all_fullnames_out
        y_out
    '''
    
    all_fullnames_out = list(np.copy(all_fullnames))
    y_out = np.copy(y)

    indx_to_keep = np.where(~np.isnan(y))
    all_fullnames_out = list(np.array(all_fullnames)[indx_to_keep])
    y_out = list(np.array(y)[indx_to_keep])

    return all_fullnames_out, y_out









