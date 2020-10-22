'''
Created on Thursday - September 10 2020, 17:38:01

@author: Michele Svanera, University of Glasgow

'''


################################################################################################################
## Imports

from __future__ import division, print_function

import os, sys
import argparse
from os.path import join as opj

import numpy as np
import nibabel as nib


################################################################################################################
## Paths and Constants

Path_in = '../data/'
Path_out = '../BIDS/'

Dim_cereb = (256, 352, 224)         # (sag, cor, lon)


################################################################################################################
## Functions

def findListOfAnatomical(path_in, identifier='nii.gz'):

    all_anat = []
    for root, _, files in os.walk(path_in):
        for i_file in files:
            if i_file.endswith(identifier):
                all_anat.append(root + '/' + i_file)
    
    return sorted(list(np.unique(all_anat)))


################################################################################################################
## Main


# Find all volumes
all_subject_path = findListOfAnatomical(Path_in)
all_subject_path = [i for i in all_subject_path if 't1w_orient-std_brain.nii.gz' in i]

if not os.path.exists(Path_out):
    os.makedirs(Path_out)
    
for i_subj in all_subject_path:

    # Load the original data
    T1 = nib.load(i_subj)
    T1_data = T1.get_data()
    # print(T1.shape)

    # Find number of slices we can save in the 1st dimension
    last_empty_slide_x1 = 0         
    for x in range(T1.shape[0]):
        if np.sum(T1_data[x,:,:]) == 0:
            last_empty_slide_x1 = x
        else:
            break
    # print(last_empty_slide_x1)

    last_empty_slide_x2 = T1.shape[0]
    for x in range(T1.shape[0]-1,0,-1):
        if np.sum(T1_data[x,:,:]) == 0:
            last_empty_slide_x2 = x
        else:
            break
    # print(last_empty_slide_x2)
    
    # Find number of slices we can save in the 2nd dimension
    last_empty_slide_y1 = 0                         # face
    for y in range(T1.shape[1]):
        if np.sum(T1_data[:,y,:]) == 0:
            last_empty_slide_y1 = y
        else:
            break

    last_empty_slide_y2 = T1.shape[1]               # back of the head
    for y in range(T1.shape[1]-1,0,-1):
        if np.sum(T1_data[:,y,:]) == 0:
            last_empty_slide_y2 = y
        else:
            break

    # Find number of slices we can save in the 3rd dimension
    last_empty_slide_z1 = 0                         # from neck
    for z in range(T1.shape[2]):
        if np.sum(T1_data[:,:,z]) == 0:
            last_empty_slide_z1 = z
        else:
            break

    last_empty_slide_z2 = T1.shape[2]               # from top of the head
    for z in range(T1.shape[2]-1,0,-1):
        if np.sum(T1_data[:,:,z]) == 0:
            last_empty_slide_z2 = z
        else:
            break

    # Cutted version
    T1_data_cut = T1_data[last_empty_slide_x1 : last_empty_slide_x2,
                          last_empty_slide_y1 : last_empty_slide_y2,
                          last_empty_slide_z1 : last_empty_slide_z2]    

    # Bring to shape = CEREBRUM shape
    
    # Compute padding to do
    x, y, z = T1_data_cut.shape
    pad_with = ((int(np.floor((Dim_cereb[0]-x)/2)),int(np.ceil((Dim_cereb[0]-x)/2))),
                (int(np.floor((Dim_cereb[1]-y)/2)),int(np.ceil((Dim_cereb[1]-y)/2))),
                (int(np.floor((Dim_cereb[2]-z)/2)),int(np.ceil((Dim_cereb[2]-z)/2))))

    
    # If the new dims are bigger, I'll cut.
    if np.sum(np.array(pad_with)<0) != 0:

        if x > Dim_cereb[0]:
            # print(x, end=' ')
            cut_with = - np.array([int(np.floor((Dim_cereb[0]-x)/2)),int(np.ceil((Dim_cereb[0]-x)/2))])
            T1_data_cut = T1_data_cut[cut_with[0] : -cut_with[1], :, :]
            
        if y > Dim_cereb[1]:
            # print(y, end=' ')
            cut_with = - np.array([int(np.floor((Dim_cereb[1]-y)/2)),int(np.ceil((Dim_cereb[1]-y)/2))])
            T1_data_cut = T1_data_cut[:, cut_with[0] : -cut_with[1], :]
            
        if z > Dim_cereb[2]:
            # print(z, end=' ')
            cut_with = - np.array([int(np.floor((Dim_cereb[2]-z)/2)), int(np.ceil((Dim_cereb[2]-z)/2))])
            T1_data_cut = T1_data_cut[:, :, cut_with[0] : -cut_with[1]]
            

    # Re-compute padding to do
    x, y, z = T1_data_cut.shape
    pad_with = ((int(np.floor((Dim_cereb[0]-x)/2)),int(np.ceil((Dim_cereb[0]-x)/2))),
                (int(np.floor((Dim_cereb[1]-y)/2)),int(np.ceil((Dim_cereb[1]-y)/2))),
                (int(np.floor((Dim_cereb[2]-z)/2)),int(np.ceil((Dim_cereb[2]-z)/2))))

    # Cutted or not, I'll pad.
    T1_data_out = np.pad(T1_data_cut, pad_with, mode='constant', constant_values=0)

    # Create output
    vol = nib.Nifti1Image(T1_data_out, affine=T1.get_sform(), header=T1.header)
    out_name = opj(os.path.dirname(i_subj), os.path.basename(i_subj)[:15] + 'T1w_cut.nii.gz')

    # Remove 'ses-1' folder and change output folder
    out_name = out_name.replace('/data/','/BIDS/')
    out_name = out_name.replace('/ses-1/','/')

    if not os.path.exists(os.path.dirname(out_name)):
        os.makedirs(os.path.dirname(out_name))

    if not os.path.exists(out_name):
        nib.save(vol, out_name)

    print(vol.shape)














