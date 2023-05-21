#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tuesday - September 08 2020, 18:25:36

@author: Michele Svanera, University of Glasgow

Script to create data augmentation for the BIDS folder.
The variables 'T1_identif' and 'GT_identif' are the volumes augmented.
The code checks if the volumes has been augmented already.

Running:
    python3 ./offline_data_augmentation.py
'''

################################################################################################################
## Imports 

from __future__ import division, print_function

import os
from os.path import join as opj
import nibabel as nib 
import numpy as np
import string

from datetime import timedelta
import time
import concurrent.futures

import elasticdeform
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
import cv2


################################################################################################################
## Paths and Constants

MAX_THREADS = 30

Input_path = '/analyse/Project0235/segmentator/data/glasgow/7T/BIDS/'
Output_path = '/analyse/Project0235/segmentator/data/glasgow/7T/BIDS_augm/'

T1_identif = 'T1w.nii.gz'
GT_identif = 'training_labels.nii.gz'

all_augmentations_geometry = ['translate', 'rotate']
all_augmentations_color = ['S&P', 'gaussian']

augm_probabil_geometry = (0.5,0.5)
augm_probabil_color = (0.5,0.5)

max_angle = 5
rot_spline_order = 3            # degrees
max_shift = [10, 15, 10]        # voxels

AUGMENTATION_FACTOR = 10        # for every anatomical 'N' volumes are created


################################################################################################################
## Functions


def roll_volume(input_vol, roll_x0, roll_x1, roll_x2):
    ''' # data augmentation - simple circular shifting '''
    

    # shift input volumes according to the "params" argument
    input_vol = np.roll(input_vol, roll_x0, axis=0)
    input_vol = np.roll(input_vol, roll_x1, axis=1)
    input_vol = np.roll(input_vol, roll_x2, axis=2)

    return input_vol


def translate_volume(input_vol, shift_x0, shift_x1, shift_x2,
                 affine=False, padding_mode = 'mirror', spline_interp_order = 0):
    '''
    # data augmentation - simple circular shifting
    # N.B. "padding_mode" and "spline_interp_order" influence the result only when "affine" is True
    '''

    # translation throught affine transformation
    if affine:

        # the padding mode is set to "wrap" by default so that the rotation through
        # affine transformation matches the circular shift through np.roll

        #padding_mode = 'constant'
        #padding_mode = 'reflect'

        M_t = np.eye(4)
        M_t[:-1, -1] = np.array([-shift_x0, -shift_x1, -shift_x2])
        
        return affine_transform(input_vol, M_t,
                                order = spline_interp_order,
                                mode = padding_mode, 
                                #constant_values = 0,
                                output_shape = input_vol.shape)

    else:
        return roll_volume(input_vol, shift_x0, shift_x1, shift_x2)


def augment(vol, gt):
    ''' Function to augment the vol with one between translation and rotation 
    plus elastic deformation.
    IN:
        vol, gt: only volume of shape [x,y,z]
    OUT:     
        vol_out, gt_out: translated or rotated + elastic deformed
    '''
    
    # Prepare the out
    vol_out = np.copy(vol)
    gt_out = np.copy(gt)
    
    # Select which transformation apply (together are slow and too distruptive)
    augmentation = np.random.RandomState().choice(all_augmentations_geometry, p=augm_probabil_geometry)
    
    ### Translation ###
    if 'translate' in augmentation:
        max_shift_x0, max_shift_x1, max_shift_x2 = max_shift           # voxels
    
        try: shift_x0 = np.random.RandomState().randint(2*max_shift_x0) - max_shift_x0
        except: shift_x0 = 0
        
        try: shift_x1 = np.random.RandomState().randint(2*max_shift_x1) - max_shift_x1
        except: shift_x1 = 0
        
        try: shift_x2 = np.random.RandomState().randint(2*max_shift_x2) - max_shift_x2
        except: shift_x2 = 0
            
        vol_out = translate_volume(vol_out, shift_x0, shift_x1, shift_x2, affine = True)
        gt_out = translate_volume(gt_out, shift_x0, shift_x1, shift_x2, affine = True)

    ### Rotation ###
    if 'rotate' in augmentation:
        random_angle = np.random.RandomState().randint(2*max_angle) - max_angle 
        rot_axes = np.random.RandomState().permutation(range(3))[:2]      # random select the 2 rotation axes
        
        vol_out = rotate(input = vol_out,
                     angle = random_angle,
                     axes = rot_axes,
                     reshape = False,
                     order = rot_spline_order,
                     mode = 'mirror',
                     prefilter = True)
        
        gt_out = rotate(input = gt_out,
                     angle = random_angle,
                     axes = rot_axes,
                     reshape = False,
                     order = 0,
                     mode = 'constant',
                     prefilter = True)
        
    ### Elastic deformation (default) ###
    rand_points = np.random.RandomState().randint(3,5)
    rand_sigma = np.random.RandomState().choice([2,3])
    vol_out, gt_out = elasticdeform.deform_random_grid([vol_out, gt_out],
                                                       sigma = rand_sigma, 
                                                       points = rand_points, 
                                                       order = [5, 0], 
                                                       mode = 'mirror')
 
    return np.clip(vol_out,0,np.inf), np.round(gt_out).astype(np.int)


def thread_function(j_augm, T1, GT, i_subj_full_id, j_sessions):

    # Perform the augmentation
    vol_augm, gt_augm = augment(T1.get_fdata(), GT.get_fdata())

    # Save result
    # j_subj_full_id = i_subj_full_id         # ex. 'sub-001' == stay the same
    j_sessions_augm = str(j_sessions).zfill(3) + j_augm

    # T1
    T1_augm = nib.Nifti1Image(vol_augm, affine=T1.affine, header=T1.header)
    t1_augm_fullpath = opj(Output_path, i_subj_full_id, 'anat', i_subj_full_id + f'_ses-{j_sessions_augm}_' + T1_identif)
    if not os.path.exists(os.path.dirname(t1_augm_fullpath)):
        os.makedirs(os.path.dirname(t1_augm_fullpath))
    nib.save(T1_augm, t1_augm_fullpath)
    
    # GT
    GT_augm = nib.Nifti1Image(gt_augm, affine=GT.affine, header=GT.header)
    gt_augm_fullpath = opj(Output_path, i_subj_full_id, 'seg', i_subj_full_id + f'_ses-{j_sessions_augm}_' + GT_identif)
    if not os.path.exists(os.path.dirname(gt_augm_fullpath)):
        os.makedirs(os.path.dirname(gt_augm_fullpath))
    nib.save(GT_augm, gt_augm_fullpath)

    # If the augmented GT or T1 is not created, then re-run the function
    if not os.path.exists(t1_augm_fullpath) or not os.path.exists(gt_augm_fullpath):
        thread_function(j_augm, T1, GT, i_subj_full_id, j_sessions)
    else:
        print(f'{j_augm} ', end='')
        
    return


################################################################################################################
## Main

start_time = time.time()    
    
# Find all GT path subject_IDs
all_subject_path = sorted(next(os.walk(Input_path))[1])
if not os.path.exists(Output_path):
    os.mkdir(Output_path)

for i_subj_full_id in all_subject_path:
    
    print(i_subj_full_id, end=': ')
    
    # Control that I didn't do it already
    all_ids_augmented = sorted(next(os.walk(Output_path))[1])
    if i_subj_full_id in all_ids_augmented:
        print('done already!')
        continue
    
    # Search for how many sessions there (looking on how many anat T1w there are)
    i_subj_root_path = opj(Input_path, i_subj_full_id)
    i_subj_all_anat_files = sorted(next(os.walk(opj(i_subj_root_path, 'anat')))[2])
    n_sessions = [i for i in i_subj_all_anat_files if T1_identif in i]

    # For loop over every session of i_subj
    for j_sessions in range(1,len(n_sessions)+1):

        # Compose fullpaths
        # i_subj_id = i_subj_full_id.split('_')[-1]
        t1_fullpath_orig = opj(i_subj_root_path, 'anat', i_subj_full_id + f'_ses-{str(j_sessions).zfill(3)}_' + T1_identif)
        gt_fullpath_orig = opj(i_subj_root_path, 'seg', i_subj_full_id + f'_ses-{str(j_sessions).zfill(3)}_' + GT_identif)
        
        # Load the original data
        T1 = nib.load(t1_fullpath_orig)
        try:
            GT = nib.load(gt_fullpath_orig)
        except:
            print('')           # Means not computed yet
            continue

        # Perform the N augmentation
        PARALLEL = min(MAX_THREADS,AUGMENTATION_FACTOR)
        with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL) as executor:
            executor.map(thread_function, list(string.ascii_lowercase)[:AUGMENTATION_FACTOR],
                        [T1]*AUGMENTATION_FACTOR,  [GT]*AUGMENTATION_FACTOR,
                        [i_subj_full_id]*AUGMENTATION_FACTOR, [j_sessions]*AUGMENTATION_FACTOR)
        
    print('')

    
print('Augmentation time: ' + str(timedelta(seconds=(time.time() - start_time))) + ' (days, hh:mm:ss.ms)\n\n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
