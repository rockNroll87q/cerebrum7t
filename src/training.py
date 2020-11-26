#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tuesday - September 08 2020, 18:12:48

@author: Michele Svanera, University of Glasgow
Training using BIDS format in input.

'''

################################################################################################################
## Imports 

from __future__ import division, print_function

import os, sys
from os.path import join as opj
import argparse
import inspect
import time
from datetime import timedelta

import numpy as np
import nibabel as nib
import json
import string

import keras
import tensorflow as tf
tf.get_logger().setLevel('INFO')        # ignore memory allocations messages

from sklearn.model_selection import train_test_split

sys.path.insert(0, '/cerebrum7t/src/')
from cerebrum7t_lib import python_utils, volume_manager, model_definition, train_test_fun, Losses3D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


################################################################################################################
## Paths and Constants

Path_in_data = '/cerebrum7t/data/BIDS/'
Path_in_data_augm = '/cerebrum7t/data/BIDS_augm/'
Path_in_meanStd = '/cerebrum7t/data/mean_std/' 
Path_out_dir = '/cerebrum7t/output/training/'


Inh_vol_path = '/cerebrum7t/data/inhomogeneity_volume.npy'

Loss_functions = list(Losses3D.losses_dict.keys())      # Import all losses

Available_models = [i for i in dir(model_definition) if 'Levels' in i]      # Import all models
models_dict = {}
for i in inspect.getmembers(model_definition, inspect.isfunction):
    models_dict[i[0]] = i[1]

Testing_volume_behavioural = ['sub-025', 'sub-045' ,'sub-075']


################################################################################################################
## Functions and Classes

def find_IDs(all_vol, root_dir):
    ''' Find the list of the subj IDs. '''

    all_filename = [os.path.basename(i) for i in all_vol]           # ex. 'sub-001_ses-001_T1w.nii.gz'
    all_vol_ID = [i.split('_')[0] for i in all_filename]            # ex. 'sub-001'

    return all_vol_ID


def searchAndAddAugmentedVolume(X_train_fullpath, Path_in_data, anat_ide, GT_to_predict,
                                augm_factor=10):
    ''' 
    Return the list of augmented volumes of the training set.
    IN:
        X_train_fullpath: full paths of training samples
        Path_in_data: expected to be the root of folders 'nii_augm' and 'GT_augm'
        anat_ide, GT_to_predict: identifiers of T1 and GT
        augm_factor: multiplication factor of training (ex. 100 times)
    OUT:
        X_train_fullpath_augm, y_train_augm: augmented training datasets
    '''

    # Find the pure IDs of the training set
    train_set_IDs = list(np.unique(find_IDs(X_train_fullpath, Path_in_data)))

    # Find the list of augmented volumes
    all_augm_T1 = volume_manager.findListOfAnatomical(Path_in_data_augm, identifier=anat_ide)

    # Keep only values which belongs to the training set
    augm_T1_train = []
    for i_vol in all_augm_T1:
        
        if np.sum([1 for j_train in train_set_IDs if j_train in i_vol]) != 0:
            augm_T1_train.append(i_vol)
        
    # Keep only a certain factor of the augmentation dataset, based on 'augm_factor'
    vol_to_select = list(string.ascii_lowercase)[ : augm_factor]
    augm_T1_train_selected = []
    for i_vol in augm_T1_train:
        if (i_vol.split(anat_ide)[0])[-2] in vol_to_select:
            augm_T1_train_selected.append(i_vol)
        
    # Find the correspondent GT
    augm_gt_train_selected = volume_manager.findCorrespondingGTPathBIDS(augm_T1_train_selected, GT_to_predict)

    if len(augm_gt_train_selected) != len(augm_T1_train_selected):
        log.info('ERROR 25.1.1: problems with data augmentation fetching: exiting...')
        sys.exit(0)
    
    return augm_T1_train_selected, augm_gt_train_selected


def split_train_test_valid(all_vol, all_GT):
    ''' Function to select unique values in order to split the datasets. '''

    # Find the IDs for all the volume in the database
    all_IDs = find_IDs(all_vol, Path_in_data)

    # Initialise vectors for set indeces
    train_indx = np.zeros((len(all_vol)))
    val_indx = np.zeros((len(all_vol)))
    test_indx = np.zeros((len(all_vol)))
    free_indx = np.ones((len(all_vol)))         # used to select free values

    # Move all behavioural test vols in testing set
    for v in Testing_volume_behavioural:
        indices = np.array([y for y, x in enumerate(all_IDs) if x == v])    # Find repetitions
        test_indx[indices] = 1
        free_indx[indices] = 0

    # Split train, val, test IDs based on their unique values
    unique_IDs = np.unique(np.array(all_IDs)[free_indx==1])
    id_train, id_testval = train_test_split(unique_IDs, test_size=0.15, random_state=42)
    id_val, id_test = train_test_split(id_testval, test_size=0.70, random_state=42)

    # Compose the datasets based on these unique selections
    for v in id_train:
        indices = np.array([y for y, x in enumerate(all_IDs) if x == v])    # Find repetitions
        train_indx[indices] = 1
        free_indx[indices] = 0
    for v in id_val:
        indices = np.array([y for y, x in enumerate(all_IDs) if x == v])    # Find repetitions
        val_indx[indices] = 1
        free_indx[indices] = 0
    for v in id_test:
        indices = np.array([y for y, x in enumerate(all_IDs) if x == v])    # Find repetitions
        test_indx[indices] = 1
        free_indx[indices] = 0

    # Create lists with train, val, test sets
    X_train = list(np.array(all_vol)[train_indx == 1])
    y_train = list(np.array(all_GT)[train_indx == 1])
    X_val = list(np.array(all_vol)[val_indx == 1])
    y_val = list(np.array(all_GT)[val_indx == 1])
    X_test = list(np.array(all_vol)[test_indx == 1])
    y_test = list(np.array(all_GT)[test_indx == 1])

    # Check the sum is consistent
    assert (np.sum(train_indx) + np.sum(val_indx) + np.sum(test_indx) + np.sum(free_indx)) == len(all_vol)
    return [X_train, X_val, X_test, y_train, y_val, y_test]
    

################################################################################################################
## Main


def main(log):
    
    ## List of every 'anat' in the database and related 'seg'
    all_volume_fullpathname = volume_manager.findListOfAnatomical(Path_in_data, identifier=anat_ide)
    all_volume_GT_fullpathname = volume_manager.findCorrespondingGTPathBIDS(all_volume_fullpathname, GT_to_predict)

    ## Load mean and std
    voxelwise_mean  = np.array(nib.load(full_path_mean_subject).get_data()).astype(dtype = 'float32')
    voxelwise_std  = np.array(nib.load(full_path_std_subject).get_data()).astype(dtype = 'float32')
    data_dims = voxelwise_mean.shape
    log.info('Volume shapes: '+ str(data_dims))
    log.info('All volumes number: ' + str(len(all_volume_fullpathname)))
    experiment_dict["data"].update({"data_dims" : data_dims})
    
    ## Split training, testing, and validation sets
    dataset_splits = split_train_test_valid(all_volume_fullpathname, all_volume_GT_fullpathname)
    X_train_fullpath, X_val_fullpath, X_test_fullpath, y_train, y_val, y_test = dataset_splits

    # Add augmented volumes
    if augmentation:
        X_train_fullpath_augm, y_train_augm = searchAndAddAugmentedVolume(X_train_fullpath, Path_in_data, anat_ide, GT_to_predict,
                                                                          augm_factor=augm_factor)
        log.info('Training set - original(#): ' + str(len(X_train_fullpath)) + ' vols.')
        log.info('Training set - augmented(#): ' + str(len(X_train_fullpath_augm)) + ' vols.')
        X_train_fullpath += X_train_fullpath_augm
        y_train += y_train_augm

    # Load the augmentation volume 'inhomogeneity_volume.npy'
    inhomogeneity_volume = np.load(Inh_vol_path)
    log.info('Loaded inhomogeneity_volume (shape): ' + str(inhomogeneity_volume.shape ))

    assert len(X_train_fullpath) == len(y_train)
    assert len(X_val_fullpath) == len(y_val)
    assert len(X_test_fullpath) == len(y_test)
    
    log.info('Training set (#): ' + str(len(X_train_fullpath)) + ' vols.')
    log.info('Validation set (#): ' + str(len(X_val_fullpath)) + ' vols.')
    log.info('Testing set (#): ' + str(len(X_test_fullpath)) + ' vols.')
    experiment_dict["data"].update({"len(X_train_fullpath)" : len(X_train_fullpath)})
    experiment_dict["data"].update({"len(X_val_fullpath)" : len(X_val_fullpath)})
    experiment_dict["data"].update({"len(X_test_fullpath)" : len(X_test_fullpath)})

    # Save CSV with list of files    
    python_utils.saveCsvWithDatasetList(X_train_fullpath, y_train, namefile=path_out_folder+'training_files.csv')
    python_utils.saveCsvWithDatasetList(X_val_fullpath, y_val, namefile=path_out_folder+'validation_files.csv')
    python_utils.saveCsvWithDatasetList(X_test_fullpath, y_test, namefile=path_out_folder+'testing_files.csv')
            
    
    ## Load validation only (actual data)
    log.info('Loading data..')
    X_val, Y_val = volume_manager.loadDatasetFromListFiles(X_val_fullpath, y_val, voxelwise_mean, voxelwise_std, data_dims)
    num_segmask_labels = Y_val.shape[-1]
    log.info('completed!\n')

    log.info('Validation set shape: ' + str(X_val.shape))
    log.info('Validation set shape (GT): ' + str(Y_val.shape))
    experiment_dict["data"].update({"X_val.shape" : X_val.shape})
    experiment_dict["data"].update({"num_segmask_labels" : num_segmask_labels})
    log.info('*** Data Loaded (val/test sets)! ***\n')
    
    
    ## Model definition
    model = models_dict[model_name](input_dim=X_val.shape[1:],
                                n_classes=num_segmask_labels, n_filters=n_filters, lr=lr, loss=loss_funct,
                                encoder_act_function=encoder_act, decoder_act_function=decoder_act,
                                use_dropout=dropout, max_n_filters=200)
                                
    model.summary(print_fn=lambda x: log.info(x), line_length=130)
    log.info('\n') 


    ## Training setup and run   
    model_filename = path_out_folder + 'model_trained.h5'
    log.info('Model will be saved at the following location: %s' % model_filename) 
    experiment_dict['model'].update({"model_filename" : model_filename})

    checkpointer = keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, verbose=1)
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=2, min_lr=1e-7, verbose=1)
    tb_callback = keras.callbacks.TensorBoard(log_dir=path_out_folder)
    if weight_class:
        class_weights = train_test_fun.computeClassWeights(Y_val)
    else:
        class_weights = [1.] * num_segmask_labels

    log.info('class_weights: ' + str(class_weights))
    train_generator = train_test_fun.TrainingGenerator(x=X_train_fullpath, y=y_train, 
                                        voxelwise_mean=voxelwise_mean, voxelwise_std=voxelwise_std,
                                        data_dims=data_dims, batch_size=batch_size, augm=augmentation,
                                        inhom_vol=inhomogeneity_volume)
    
    ## Run training
    start_time = time.time()    
    log.info('Training... %d epochs, %d vol.s per batch.\n' % (epochs, batch_size))
    history = model.fit_generator(generator = train_generator, 
                                  validation_data = (X_val, Y_val),
                                  steps_per_epoch = np.ceil(len(y_train) / batch_size),
                                  epochs = epochs,
                                  verbose = 1,
                                  callbacks = [checkpointer, earlystopper, reduce_lr, tb_callback],#, tb_callback],
                                  class_weight = class_weights,
                                  max_queue_size = 10,      # Maximum size for the generator queue
                                  workers = 5,
                                  use_multiprocessing = False)

    log.info('Training time: ' + str(timedelta(seconds=(time.time() - start_time))) + ' (days, hh:mm:ss.ms)\n\n')
   
    ## Testing: Plot and save results
    # Load testing data
    X_test, Y_test = volume_manager.loadDatasetFromListFiles(X_test_fullpath, y_test, voxelwise_mean, voxelwise_std, data_dims)
    log.info('Testing set shape: ' + str(X_test.shape))
    log.info('Testing set shape (GT): ' + str(Y_test.shape))
    experiment_dict["data"].update({"X_test.shape" : X_test.shape})

    # Plot (training) history
    history_dict = history.history
    plt.title('history_dict_training')
    plt.xlabel('epochs')
    _ = [plt.plot(history_dict[i],label=i,alpha=0.7,linewidth=2.) for i in history_dict.keys() if i!='lr']
    plt.legend()

    filename_plot = path_out_folder + 'history_dict_training.pdf'
    plt.savefig(filename_plot)
    log.info('Saved plot in: ' + filename_plot)
    plt.close()
    
    # Testing (numerically)
    log.info('*** Testing... ***')

    train_test_fun.testModel(model, X_val, Y_val, 'Validation set')
    train_test_fun.testModel(model, X_test, Y_test, 'Testing set')
    
    # Test saving out volumes: Export volume and probability maps
    log.info('Test and save volumes')
    train_test_fun.testModelAndSaveOutVolumes(model, X_test, Y_test, X_test_fullpath, path_out_folder)

    # Save experiment details and history
    with open(json_filename, 'w') as j_f:
        json.dump(experiment_dict, j_f, indent=4, ensure_ascii=False)

    for j_key in history_dict.keys():           # Convert to string every number to make it JSON serializable
        history_dict[j_key] = list(str(i) for i in history_dict[j_key])
    with open(json_history_filename, 'w') as j_f:
        json.dump(history_dict, j_f, indent=4, ensure_ascii=False)
        
    return


if __name__ == "__main__": 

    # Check if there are enough resources to train the model
#    if python_utils.checkGPUsAvailability(n_gpus=4) != True:
#        sys.exit(0)
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Classifier training')
    
    parser.add_argument('--model', default='ThreeLevelsConvUNetStridedConv', choices=models_dict.keys(), help='Model name')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate (float)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size (int)')
    parser.add_argument('--epochs', type=int, default=100, help='epochs (int)')
    parser.add_argument('--n_filters', default=24, type=int, help='number of filters for the first layer (double each layer)')   
    parser.add_argument('--loss_funct', default='categorical_crossentropy', choices=Loss_functions, help='loss function')
    parser.add_argument('--encoder_act_funct', default='relu', choices=['relu','elu'], help='Activation function encoder.')
    parser.add_argument('--decoder_act_funct', default='relu', choices=['relu','elu'], help='Activation function encoder.')
    parser.add_argument('--dropout', action='store_true', help='use dropout (add flag to use it)')
    parser.add_argument('--weight_class', action='store_true', help='weight the class numerosity (add flag to use it)')

    parser.add_argument('--anat_ide', default='T1w', type=str, help='T1w identifier (str)')
    parser.add_argument('--GT_to_predict', default='training_labels', type=str, help='GT identifier (str)')
    parser.add_argument('--augmentation', action='store_true', help='use augmentation (add flag to use it)')
    parser.add_argument('--augm_factor', type=int, default=100, help='factor of: how many times the training set use for augmentation')

    args = parser.parse_args()
        
    # Model args
    model_name = str(args.model)
    lr = float(args.learning_rate)
    batch_size = args.batch_size
    epochs = args.epochs
    n_filters = args.n_filters
    loss_funct = args.loss_funct
    encoder_act = args.encoder_act_funct
    decoder_act = args.decoder_act_funct
    dropout = args.dropout
    weight_class = args.weight_class
    
    # Data args    
    anat_ide = args.anat_ide + '.nii.gz'
    GT_to_predict = args.GT_to_predict + '.nii.gz'
    augmentation = args.augmentation
    augm_factor = args.augm_factor
        
    # Paths of mean/std volumes
    full_path_mean_subject = opj(Path_in_meanStd, '_' + args.anat_ide + '_average_subject.nii.gz')
    full_path_std_subject = opj(Path_in_meanStd, '_' + args.anat_ide + '_stdDev_subject.nii.gz')
        
    # Initialisation
    exper_description = 'anatIde=' + args.anat_ide.replace('_','-') + '_GT=' + \
                        (args.GT_to_predict).replace('_','-') + '_nf=' + str(n_filters) + \
                        '_epoch=' + str(epochs) + '_BIDS'
    if augmentation:
        exper_description += '_augm'
    path_out_folder = python_utils.initialiseOutputFolder(Path_out_dir,log_description=exper_description)     # Create out folder
    print('folder: ' + str(path_out_folder.split('/')[-2]))
    Log_filename = path_out_folder + 'training.log'    
    log = python_utils.initialiseLogger(Log_filename, log_level=20)     # 20: INFO
    python_utils.logEveryPackageLoad(log)
    
    # Save a json file with every useful info about the training (paths, model, etc.)
    json_filename = path_out_folder + 'experiment_detail.json'
    json_history_filename = path_out_folder + 'training_history.json'
    experiment_dict = {}
    experiment_dict['experiment_name'] = path_out_folder.split('/')[-2]
    experiment_dict['path'] = {"path_out_folder" : path_out_folder,
                                "Log_filename" : Log_filename,
                                "Path_in_data" : Path_in_data,
                                "full_path_mean_subject" : full_path_mean_subject,
                                "full_path_std_subject" : full_path_std_subject
                                }
    experiment_dict['model'] = {"model": model_name,
                                "lr" : lr,
                                "batch_size" : batch_size,
                                "epochs" : epochs,
                                "n_filters" : n_filters,
                                "loss_funct" : loss_funct,
                                "encoder_act" : encoder_act,
                                "decoder_act" : decoder_act
                                }
    experiment_dict['data'] = {"anat_ide" : anat_ide,
                               "GT_to_predict" : GT_to_predict,
                               "use_augmentation" : augmentation,
                               "augm_factor" : augm_factor
                               }
    
    # Log experiment info
    log.info("############### SELECTED PARAMETERS ###############")
    for arg in vars(args):
        log.info((arg, getattr(args, arg)))
    log.info('full_path_mean_subject: ' + full_path_mean_subject)
    log.info('full_path_std_subject: ' + full_path_std_subject)
    log.info('path_out_folder: ' + path_out_folder)
    log.info('Log_filename: ' + Log_filename)
    log.info('json_filename: ' + json_filename)
    log.info('json_history_filename: ' + json_history_filename)
    log.info("#" * 50 + '\n')

    main(log)








# Model args: testing arguments
# lr = 0.001
# batch_size = 1
# epochs = 2
# n_filters = 20
# loss_funct = 'categorical_crossentropy'
# encoder_act = 'relu'
# decoder_act = 'relu'
# dropout = True
# weight_class = True
# model_name='ThreeLevelsConvUNetStridedConv'

# # Data args
# anat_ide = 'T1w.nii.gz'
# GT_to_predict = 'training_labels.nii.gz'
# augmentation = True
# augm_factor = 10

# full_path_mean_subject = opj(Path_in_meanStd, '_T1w_average_subject.nii.gz')
# full_path_std_subject = opj(Path_in_meanStd, '_T1w_stdDev_subject.nii.gz')
   
# exper_description = 'anatIde=' + '_T1w'.replace('_','-') + '_GT=' + ('Fracasso16-mc').replace('_','-') + '_nf=' + str(n_filters) + '_epoch=' + str(epochs) + '_BIDS'


