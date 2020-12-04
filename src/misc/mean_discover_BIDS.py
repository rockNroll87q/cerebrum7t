'''
Created on Wednesday - September 09 2020, 13:43:48

@author: Michele Svanera, University of Glasgow

Code to calcolate the mean and the variance across the training set of anatomical 

'''


################################################################################################################
## Imports 

from __future__ import division, print_function

import os
import argparse

import numpy as np
import nibabel as nib


################################################################################################################
## Paths and Constants

T1_IDENTIFIERS = ['_T1w.nii.gz',
                '_INV1.nii.gz',
                '_INV1.nii.gz']

Raw_path = '/path/to/BIDS/'
Out_path =  '/path/to/mean_std/'


################################################################################################################
## Functions


def findListOfAnatomical(path_in, identifier='nii.gz'):

    all_anat = []
    for root, _, files in os.walk(path_in):
        for file in files:
            if file.endswith(identifier):
                all_anat.append(root + '/' + file)
    
    return sorted(all_anat)



################################################################################################################
## Main


def main():
    
    all_subject_filename = findListOfAnatomical(Raw_path, identifier=anat_ide)
    
    if len(all_subject_filename) == 0:
        print('Error: file not found.')
        return
    
    # Allocate variable
    subject_zero = nib.load(all_subject_filename[0])
    data_dims = subject_zero.shape
    average_subject = np.zeros((data_dims))
    std_dev_subject = np.zeros((data_dims))
    var_subject     = np.zeros((data_dims))
    print(data_dims)
    
    N_subjects = len(all_subject_filename)
    
    # Compute mean
    print("Loading... ")
    subj_num = 0
    for i_subj_full_path in all_subject_filename:
        
        if subj_num % 50 == 0:
            print('Subj %3d/%3d'%(subj_num, N_subjects))
    
        # Load i anat
        i_anat_data = nib.load(i_subj_full_path)
        i_anat_data_loaded = i_anat_data.get_data()
        
        try:
            average_subject += i_anat_data_loaded / float(N_subjects)
        except:
            dims_scan = "(" + "".join([str(x)+" " for x in i_anat_data_loaded.shape])[:-1] + ")"
            print("Subject -" + os.path.basename(i_subj_full_path) + "- has dim: " + dims_scan + " - FullPath: " + i_subj_full_path)
    
        subj_num += 1
        
    # Save mean
    img_to_save = nib.Nifti1Image(average_subject, header=i_anat_data.header, affine=i_anat_data.affine)
    nib.save(img_to_save, Out_path+anat_ide[:-7]+'_average_subject.nii.gz')
    
    print("Average computed! Moving on to the standard deviation...\n")
    
    
    # Compute standard deviation 
    print("Loading... ")
    subj_num = 0
    for i_subj_full_path in all_subject_filename:
        
        if subj_num % 50 == 0:
            print('Subj %3d/%3d'%(subj_num, N_subjects))
    
        # Load i anat
        i_anat_data = nib.load(i_subj_full_path)
        i_anat_data_loaded = i_anat_data.get_data()
        
        try:
            # apply the definition of standard deviation, i.e. std = sqrt( 1/N * sum( (x - u_x)^2 ) )
            # (compute the sqrt after the normalised-by-N sum...)
            var_subject += np.square( i_anat_data_loaded - average_subject ) / float(N_subjects)
        except:
            dims_scan = "(" + "".join([str(x)+" " for x in i_anat_data_loaded.shape])[:-1] + ")"
            print("Subject -" + os.path.basename(i_subj_full_path) + "- has dim: " + dims_scan + " - FullPath: " + i_subj_full_path)
    
        subj_num += 1
    
    std_dev_subject = np.sqrt(var_subject)
    
    # Save std
    img_to_save = nib.Nifti1Image(std_dev_subject, header=i_anat_data.header, affine=i_anat_data.affine)
    nib.save(img_to_save, Out_path+anat_ide[:-7]+'_stdDev_subject.nii.gz')
    
    print("Standard deviation computed!\n")



if __name__ == "__main__": 

    # Argument parsing
    parser = argparse.ArgumentParser(description='Classifier training')
    
    parser.add_argument('--anat_ide', choices=T1_IDENTIFIERS, type=str, help='T1w identifier (str)')
                        
    args = parser.parse_args()
        
    # Model args
    anat_ide = args.anat_ide
    
    main()












    
