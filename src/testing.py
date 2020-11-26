# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:43:50 2019

@author: Michele Svanera, Dennis Bontempi
University of Glasgow.

Code to perform the testing.
"""


################################################################################################################
## Imports 

from __future__ import division, print_function

import os, sys
import argparse
import time
from datetime import timedelta

import numpy as np
import nibabel as nib
import json
from scipy.ndimage import morphology

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import load_model

sys.path.insert(0, '/cerebrum7t/src/')
from cerebrum7t_lib import Losses3D


################################################################################################################
## Paths and Constants

Path_in_models = '/cerebrum7t/output/training/'
Path_in_meanStd = '/cerebrum7t/data/mean_std/'
Path_out_dir = '/cerebrum7t/output/testing/'
Testing_volume = '/cerebrum7t/data/testing_volume.nii.gz'
Loss_functions = Losses3D.losses_dict

#Classes_labels = ['background',         #class = 0
#                'Gray matter',          #class = 1 
#                'basic ganglia',        #class = 2
#                'White matter',         #class = 3
#                'ventricles',           #class = 4
#                'cerebellum',           #class = 5
#                'brainstem']            #class = 6

Error_subjects_not_found = "Subjects provided not found (check bids paths or subjects)"


################################################################################################################
## Functions and Classes


def volumeZscoring(input_vol, voxelwise_mean, voxelwise_std):
    '''
    Function to z-scoring of loaded-by-generator data 
    '''
    
    # standardize each training volume
    input_vol -= voxelwise_mean
    
    # Prevent division by zero
    voxelwise_std[voxelwise_std == 0] = 1
    input_vol /= voxelwise_std
    
    return input_vol



################################################################################################################
## Main


def main():

    # Prepare for testing    
    start_time = time.time()
    data_dims = experiment_dict['data']['data_dims']        # (sag, cor, lon)

    # Load model
    print('\n\nLoading the model: ', end='')
    partial_start_time = time.time()
    model = load_model(os.path.join(path_model, 'model_trained.h5'), custom_objects=Loss_functions)
    print(str(timedelta(seconds=(time.time() - partial_start_time))) + ' (hh:mm:ss.ms)')
    
    # Load mean and std
    print('Loading mean/std volumes: ', end='')
    partial_start_time = time.time()
    mean_fullpath = Path_in_meanStd + os.path.basename(experiment_dict['path']['full_path_mean_subject']) 
    std_fullpath = Path_in_meanStd + os.path.basename(experiment_dict['path']['full_path_std_subject'])
    voxelwise_mean = np.array(nib.load(mean_fullpath).get_data()).astype(dtype = 'float32')
    voxelwise_std  = np.array(nib.load(std_fullpath).get_data()).astype(dtype = 'float32')

    ## Create an output folder
    path_out_folder = os.path.join(Path_out_dir, 'testing/')
    if not os.path.exists(path_out_folder):
        os.makedirs(path_out_folder)
    print(str(timedelta(seconds=(time.time() - partial_start_time))) + ' (hh:mm:ss.ms)')
            
    ## Located and load the full volume
    print('Loading the testing volume: ', end='')
    partial_start_time = time.time()
    if not os.path.exists(testing_volume):
        raise Exception('T1w not found')
    t1_full = nib.load(testing_volume)
    t1_full_vol = t1_full.get_data()

    assert list(t1_full.shape) == experiment_dict['data']['data_dims'], "T1w has a different shape than training volumes"
        
    ## Load the data and prepare            
    # Prepare testing data
    X_test = volumeZscoring(t1_full_vol.astype(dtype='float32'), voxelwise_mean, voxelwise_std)
    X_test = X_test.reshape((data_dims[0], data_dims[1], data_dims[2], 1))
    X_test = np.expand_dims(X_test, axis=0)
    print(str(timedelta(seconds=(time.time() - partial_start_time))) + ' (hh:mm:ss.ms)')

    ## Testing
    print('Inference: ', end='')
    partial_start_time = time.time()
    y_test_prob_map = np.squeeze(model.predict(X_test, verbose=False))

    # Compute the argmag (hard-thresholding) of the proabability map to obtain the labelled volume
    y_test_pred = np.argmax(y_test_prob_map, axis=-1).astype(dtype='uint8')
    print(str(timedelta(seconds=(time.time() - partial_start_time))) + ' (hh:mm:ss.ms)')

    # Save output (pred and prob maps)
    print('Saving: ', end='')
    partial_start_time = time.time()
    pred_vol = nib.Nifti1Image(y_test_pred, affine=t1_full.get_sform(), header=t1_full.header)
    pred_vol_name = os.path.join(path_out_folder, 'predicted.nii.gz')
    nib.save(pred_vol, pred_vol_name)

    prob_map_vol = nib.Nifti1Image(y_test_prob_map, affine=t1_full.get_sform(), header=t1_full.header)
    prob_map_vol_name = os.path.join(path_out_folder, 'probability_maps.nii.gz')
    nib.save(prob_map_vol, prob_map_vol_name)

    # Create the brain mask and save it
    brain_mask = morphology.binary_dilation((y_test_pred > 0).astype('float'), iterations=5)
    brain_masked = np.copy(t1_full_vol) * brain_mask
        
    # Save output
    brain_masked_vol = nib.Nifti1Image(brain_masked, affine=t1_full.get_sform(), header=t1_full.header)
    brain_masked_name = os.path.join(path_out_folder ,'brain_masked.nii.gz')
    nib.save(brain_masked_vol, brain_masked_name)
    
    print(str(timedelta(seconds=(time.time() - partial_start_time))) + ' (hh:mm:ss.ms)')
    print('\n** Total time: ' + str(timedelta(seconds=(time.time() - start_time))) + ' (hh:mm:ss.ms) **\n')
        
    return


if __name__ == "__main__": 

    # Argument parsing
    parser = argparse.ArgumentParser(description='Classifier testing on BIDS')

    parser.add_argument('--training_name', type=str, help='Starting model', required=True)
    args = parser.parse_args()
        
    # Model args
    training_name = args.training_name
    testing_volume = Testing_volume
    
    # Initialisation
    path_model = os.path.join(Path_in_models, training_name)
    with open(os.path.join(path_model, 'experiment_detail.json')) as f:
        experiment_dict = json.load(f)

    main()




