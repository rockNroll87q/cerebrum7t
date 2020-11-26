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
import gc

import numpy as np
import nibabel as nib

import keras
import sklearn as skl
import cv2

sys.path.insert(0, '../../')
from cerebrum7t_lib import volume_manager


################################################################################################################
## Paths and Constants

all_augmentations_color = ['S&P', 'gaussian', 'inhomogeneity_noise']
augm_probabil_color = (1/4,1/4,1/2)        #(0.33,0.33,0.33)


################################################################################################################
## Functions


def AugmentationSaltAndPepperNoise(X_data, amount=5./1000):
    ''' Function to add S&P noise to the volume.
    IN:
        input volume (3D)
        amount (default=5/1000): Quantity of voxels affected
    OUT:
        output volume
    '''

    X_data_out = X_data
    salt_vs_pepper = 0.2        # Ration between salt and pepper voxels
    n_salt_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * salt_vs_pepper))
    n_pepper_voxels = int(np.ceil(amount * np.prod(X_data_out.size) * (1.0 - salt_vs_pepper)))
    
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(n_salt_voxels)) for i in X_data.shape[:3]]
    X_data_out[coords[0],coords[1],coords[2]] = np.max(X_data)
        
    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(n_pepper_voxels)) for i in X_data.shape[:3]]
    X_data_out[coords[0],coords[1],coords[2]] = np.min(X_data)
    
    return X_data_out


def AugmentationGaussianNoise(X_data):
    ''' Function to add gaussian noise to the volume.
    IN:
        input volume (3D)
    OUT:
        output volume
    '''
    
    # Gaussian distribution parameters
    X_data_no_background = X_data#[X_data > 100]
    mean = np.mean(X_data_no_background)
    var = np.var(X_data_no_background)
    sigma = var ** 0.5
    
    gaussian = np.random.normal(mean, sigma, X_data.shape).astype(X_data.dtype)
    
    # Compose the output (src1, alpha, src2, beta, gamma)
    X_data_out = cv2.addWeighted(X_data, 0.8, gaussian, 0.2, 0)    
    
    return X_data_out


def AugmentationInhomogeneityNoise(X_data, inhom_vol):
    ''' Function to add inhomogeneity noise to the volume.
    IN:
        input volume (3D)
        inhomogeneity_volume (3D)
    OUT:
        output volume
    '''

    # Randomly select a vol of the same shape of 'X_data'
    x_1 = np.random.randint(0, int(X_data.shape[0])-1, size=1)[0]
    x_2 = np.random.randint(0, int(X_data.shape[1])-1, size=1)[0]
    x_3 = np.random.randint(0, int(X_data.shape[2])-1, size=1)[0]
    y_1 = inhom_vol[x_1 : x_1 + X_data.shape[0], 
                    x_2 : x_2 + X_data.shape[1], 
                    x_3 : x_3 + X_data.shape[2]]
    
    return X_data + y_1[:,:,:,np.newaxis]              # Add noise to the original vol


def augment(vol, gt, inhom_vol):
    ''' Function to augment the vol with one between S&P noise or gaussian
    IN:
        vol, gt: only volume of shape [x,y,z]
    OUT:     
        vol_out, gt 
    '''
    
    # Prepare the out
    vol_out = np.copy(vol)
    
    ### Color transformation ###
    augmentation = np.random.choice(all_augmentations_color, p=augm_probabil_color)
    
    # Always apply an augmentation
    if 'S&P' in augmentation:
        vol_out = AugmentationSaltAndPepperNoise(vol_out)

    if 'gaussian' in augmentation:
        vol_out = AugmentationGaussianNoise(vol_out)

    if 'inhomogeneity_noise' in augmentation:
        vol_out = AugmentationInhomogeneityNoise(vol_out, inhom_vol)

    return vol_out, gt


class TrainingGenerator(keras.utils.Sequence):
    '''
    Sequence are a safer way to do multiprocessing. 
    This structure guarantees that the network will only train once on each 
    sample per epoch which is not the case with generators.
    Followed tutorial from: 
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Function that creates a generator that handles data loading dynamically 
    in order to fit the entire dataset into RAM
    IN:
        x: list with every full path filename of training set.
        y: GT of the volumes
        data_dims: 3D data shape of volumes
        voxelwise_mean, voxelwise_std: volumes with DB's mean and std
        n_classes: GT data to check everything match
        batch_size: batch_size
    OUT:
        return batch data
    '''

    def __init__(self, x, y, data_dims, voxelwise_mean, voxelwise_std, inhom_vol,
                 num_labels=0, batch_size=1, augm=True, shuffle=True):
        # Initialization
        self.x, self.y = x, y
        self.data_dims = data_dims
        self.voxelwise_mean, self.voxelwise_std = voxelwise_mean, voxelwise_std
        self.inhom_vol = inhom_vol
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augm = augm

    def __len__(self):
        # Number of batches per epoch.
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):

        gc.collect()
        # Define the batch to load [x,y]
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        # Load the actual data
        X_batch, Y_batch = self.__data_generation(batch_x, batch_y)
        gc.collect()
        
        return X_batch, Y_batch
    
    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle == True:
            # skl.utils.shuffle the list of training samples
            self.x, self.y = skl.utils.shuffle(self.x, self.y)
        
    def __data_generation(self, batch_x, batch_y):
        # Load actual volumes: T1 and segmentation mask
        x_train_batch, y_train_batch = volume_manager.loadDatasetFromListFiles(batch_x, batch_y, 
                                                                               self.voxelwise_mean, 
                                                                               self.voxelwise_std, 
                                                                               self.data_dims,
                                                                               self.num_labels)
        
        if self.augm:
            # For every sample in the batch
            x_train_batch_augm = []
            y_train_batch_augm = []
            for i_batch in range(self.batch_size):
                
                i_batch_augm_x, i_batch_augm_y = augment(x_train_batch[i_batch,], 
                                                         y_train_batch[i_batch,],
                                                         self.inhom_vol)
                
                x_train_batch_augm.append(i_batch_augm_x)
                y_train_batch_augm.append(i_batch_augm_y)
            
            x_train_batch_augm = np.stack(x_train_batch_augm,axis=0)
            y_train_batch_augm = np.stack(y_train_batch_augm,axis=0)
        else:
            x_train_batch_augm, y_train_batch_augm = x_train_batch, y_train_batch

        return x_train_batch_augm, y_train_batch_augm


def computeClassWeights(Y_val):
    ''' Compute class weights to feed the fit function. '''
    y_val = np.argmax(Y_val, axis=-1)
    classes = np.unique(y_val)
    weigths = skl.utils.class_weight.compute_class_weight('balanced',classes,y_val.flatten())
    return weigths


def testModel(model, X_testval, Y_testval, identifier, batch_size=1, wand=False):
    ''' Function to test the model but only numerically
    '''
    score = model.evaluate(X_testval, Y_testval, verbose=False, batch_size=batch_size)
        
    _ = [log.info(identifier + ' (' + i_metric + '): ' + str(i_score)) for i_metric, i_score in zip(model.metrics_names,score)]

    if wand:
        import wandb
        metrics = {}
        for i_metric, i_score in zip(model.metrics_names,score):
            metrics[identifier + '_' + str(i_metric)] = i_score
        wandb.log(metrics)

    return 


def dynamicTestModel(model, log, generator, identifier):
    ''' Function to dynamically test the model (numerically)
    '''
    score = model.evaluate_generator(generator, verbose=1, max_queue_size = 10, use_multiprocessing = False)

    _ = [log.info(identifier + ' (' + i_metric + '): ' + str(i_score)) for i_metric, i_score in zip(model.metrics_names,score)]

    return 


def testModelAndSaveOutVolumes(model, X_testval, Y_testval, X_testval_fullpath, path_out_folder):
    ''' Function to test the model saving probability maps and output GT
    '''
    
    # Create out folder for testing
    out_dir = path_out_folder + 'testing/'
    try:
        os.mkdir(out_dir)
    except Exception as e:
        print(e)
    
    # For every volume in testVal set
    for i_test in range(X_testval.shape[0]):

        # Compute the out value from the model
        i_vol_test = np.expand_dims(X_testval[i_test,], axis=0)
        y_test_prob_map = np.squeeze(model.predict(i_vol_test, verbose=False))

        # compute the argmag (hard-thresholding) of the proabability map to obtain the labelled volume
        y_test_pred = np.argmax(y_test_prob_map, axis=-1).astype(dtype='uint8')

        # Compose the headers
        i_fullfilename_in = X_testval_fullpath[i_test]
        i_vol = nib.load(i_fullfilename_in)
        vol_header = i_vol.header
        
        # Compose the output filename 
        i_subj_id = os.path.basename(i_fullfilename_in)[:-7]          # ex. 'sub-005_ses-001_T1w'
        i_fullfilename_out = opj(out_dir, i_subj_id + '.nii.gz')
        i_fullfilename_gt = opj(out_dir, i_subj_id + '_gt.nii.gz')
        i_fullfilename_pred = opj(out_dir, i_subj_id + '_predicted_volume.nii.gz')
        i_fullfilename_prob = opj(out_dir, i_subj_id + '_prob_map_volume.nii.gz')

        # Save out files
        nib.save(i_vol, i_fullfilename_out)

        y_gt = np.argmax(Y_testval[i_test,],axis=-1)
        gt_vol = nib.Nifti1Image(y_gt, affine=vol_header.get_sform(), header=vol_header)
        nib.save(gt_vol, i_fullfilename_gt)

        pred_vol = nib.Nifti1Image(y_test_pred, affine=vol_header.get_sform(), header=vol_header)
        nib.save(pred_vol, i_fullfilename_pred)

        prob_map_vol = nib.Nifti1Image(y_test_prob_map, affine=vol_header.get_sform(), header=vol_header)
        nib.save(prob_map_vol, i_fullfilename_prob)
            
    return
          




