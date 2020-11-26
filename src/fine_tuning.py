#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tuesday - September 08 2020, 18:12:48

@author: Michele Svanera, University of Glasgow
Fine-tuning the model.

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
import pandas as pd
import nibabel as nib
import json
import string

import keras
import tensorflow as tf
tf.get_logger().setLevel('INFO')        # ignore memory allocations messages
from keras.models import load_model
from keras.layers.convolutional import Conv3D
from keras.models import Model

# from keras import backend as K
import gc

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
Path_out_dir = '/cerebrum7t/output/fine_tuning/'
Path_in_models = '/cerebrum7t/output/training/'

# Training folder name
Inh_vol_path = '/cerebrum7t/data/inhomogeneity_volume.npy'

# File with indications of training volumes
Training_samples_file = '/cerebrum7t/data/training_samples.csv'

Loss_functions = Losses3D.losses_dict                                       # Import all losses

Available_models = [i for i in dir(model_definition) if 'Levels' in i]      # Import all models
models_dict = {}
for i in inspect.getmembers(model_definition, inspect.isfunction):
    models_dict[i[0]] = i[1]



################################################################################################################
## Functions and Classes

def find_IDs(all_vol, root_dir):
    ''' Find the list of the subj IDs. '''

    all_filename = [os.path.basename(i) for i in all_vol]           # ex. 'sub-001_ses-001_T1w.nii.gz'
    all_vol_ID = [i.split('_')[0] for i in all_filename]            # ex. 'sub-001'

    return all_vol_ID


def loadAndSetTrainingSamples():
    ''' Load a csv where I decided which samples to use for training '''

    df = pd.read_csv(Training_samples_file)
    training_vol = df[df['training set?'] == 'T']
    all_subj_ID = ['sub-'+f"{int(i):03d}" for i in list(training_vol['Volume'])]

    return all_subj_ID


def split_train_test_valid(all_vol, all_GT):
    ''' Function to select unique values in order to split the datasets. 
        Rationate: there is only 1 session per subj. 
        Which means we can just select subjects.
    '''

    # Retrieve training samples
    id_train = loadAndSetTrainingSamples()                                  # 20 x train

    # Find the IDs for all the volume in the database ('augm' folder)
    all_IDs = find_IDs(all_vol, Path_in_data)

    # Split train, val, test IDs based on their unique values
    id_testval = sorted(list(set(all_IDs) - set(id_train)))
    id_val, id_test = train_test_split(id_testval, test_size=0.80, random_state=42)     # ~15 x val

    # Create lists with train, val, test sets
    X_train = [i for i in all_vol for j_train in id_train if j_train in i]
    y_train = [i for i in all_GT  for j_train in id_train if j_train in i]
    X_val   = [i for i in all_vol for j_val in id_val     if j_val in i]
    y_val   = [i for i in all_GT  for j_val in id_val     if j_val in i]
    X_test  = [i for i in all_vol for j_test in id_test   if j_test in i]
    y_test  = [i for i in all_GT  for j_test in id_test   if j_test in i]

    # Add to training the augm volumes
    all_augm_vol = volume_manager.findListOfAnatomical(Path_in_data_augm, identifier=anat_ide)
    all_augm_vol = [i for i in all_augm_vol for j in id_train if j in i]        # keep only training vol.s
    all_augm_vol_gt = volume_manager.findCorrespondingGTPathBIDS(all_augm_vol, GT_to_predict)
    all_augm_vol, all_augm_vol_gt = removeMissingVolumes(all_augm_vol, all_augm_vol_gt)

    return [X_train + all_augm_vol, X_val, X_test, y_train + all_augm_vol_gt, y_val, y_test]
    

def removeMissingVolumes(all_volume_fullpathname, all_volume_GT_fullpathname):
    """ Check that at every anatomical correspond a seg volume
    Args:
        all_volume_fullpathname ([list]): initial list of vol
        all_volume_GT_fullpathname ([list]): initial list of gt
    Returns:
        [list]: lists of vol and gt
    """

    all_vol = []; all_gt = []

    # Keep only vol and gt pairs that exists
    for i_vol, i_gt in zip(all_volume_fullpathname, all_volume_GT_fullpathname):
        if os.path.exists(i_vol) and os.path.exists(i_gt):
            all_vol.append(i_vol)
            all_gt.append(i_gt)

    return all_vol, all_gt


################################################################################################################
## Main


def main(log, my_loss_funct):
    
    ## List of every 'anat' in the database and related 'seg'
    all_volume_fullpathname = volume_manager.findListOfAnatomical(Path_in_data, identifier=anat_ide)
    all_volume_GT_fullpathname = volume_manager.findCorrespondingGTPathBIDS(all_volume_fullpathname, GT_to_predict)

    all_volume_fullpathname, all_volume_GT_fullpathname = removeMissingVolumes(all_volume_fullpathname, all_volume_GT_fullpathname)

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
    

    ## Rationale: it's not easily possible to load directly an old model and fine tune it with multi-gpus.
    # What we do here is to create a new model with the same architecture of the old one.
    # The weights of this new model are random.
    # We then load the old model and copy the old weights into the new one.

    # Load old experiment in order to give proper value to the model
    with open(opj(Path_in_models, training_name, 'experiment_detail.json')) as f:
        old_experiment_dict = json.load(f)
    
    ## Model definition
    if my_loss_funct == 'NaN':                                     # If not specified, copy the old loss function
        my_loss_funct = old_experiment_dict['model']['loss_funct']

    model_name = old_experiment_dict['model']['model']          # ex. 'ThreeLevelsConvUNetStridedConv'
    model = models_dict[model_name](input_dim=X_val.shape[1:],
                                    n_classes=num_segmask_labels, 
                                    n_filters=old_experiment_dict['model']['n_filters'], 
                                    lr=lr,              # new LR
                                    loss=my_loss_funct,#,
                                    use_dropout=dropout,
                                    encoder_act_function=old_experiment_dict['model']['encoder_act'], 
                                    decoder_act_function=old_experiment_dict['model']['decoder_act'])

    # Load the old model (on CPU)
    with tf.device('/cpu:0'):
        base_model = load_model(opj(Path_in_models, training_name, 'model_trained.h5'), custom_objects=Loss_functions)
    log.info('** BASE MODEL: **') 
    base_model.summary(print_fn=lambda x: log.info(x), line_length=130)
    log.info('\n') 

    # Copy conv layer weights for the old model into the new one
    layer_names_to_update = [i.name for i in model.layers if '_conv' in i.name or 'main' in i.name] 
    layer_names_to_update = layer_names_to_update[:-last_layer]
    log.info('Kept layers:')
    log.info(layer_names_to_update)

    # Copy layers from old (base_)model to the new model 
    for i_layer in layer_names_to_update:
        if 'main' not in i_layer:                                   # not so much sense copy those
            i_layer_old = base_model.get_layer(i_layer).get_weights()
            model.get_layer(i_layer).set_weights(i_layer_old)
            del i_layer_old

    # Delete old model to save space
    del base_model
    gc.collect()
 
    ## Training setup and run  
    def train_my_net(model, fine_tuning, final_model_name='model_trained.h5'):
        """
        Define the function performing the two steps of training: Warming-up and Fine tuning.
        """

        # Log model details
        log.info('** NEW MODEL: **') 
        model.summary(print_fn=lambda x: log.info(x), line_length=130)
        log.info('\n') 

        # Define model metadata and callbacks
        model_filename = path_out_folder + final_model_name
        log.info('Model will be saved at the following location: %s' % model_filename) 
        experiment_dict['model'].update({"model_filename" : model_filename})
        checkpointer = keras.callbacks.ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, verbose=1)
        earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, min_lr=1e-7, verbose=1)

        tb_callback = keras.callbacks.TensorBoard(log_dir=path_out_folder)#+'TensorBoard')
        if weight_class:
            class_weights = train_test_fun.computeClassWeights(Y_val)
        else:
            class_weights = [1.] * num_segmask_labels

        log.info('class_weights used:')
        log.info(class_weights)
        train_generator = train_test_fun.TrainingGenerator(x=X_train_fullpath, y=y_train, 
                                            voxelwise_mean=voxelwise_mean, voxelwise_std=voxelwise_std,
                                            data_dims=data_dims, num_labels=num_segmask_labels, augm=False,
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

        return history


    def plot_and_save_history(history, out_name='warming_up'):

        # Plot (training) history
        history_dict = history.history
        plt.title('history_dict_training')
        plt.xlabel('epochs')
        _ = [plt.plot(history_dict[i],label=i,alpha=0.7,linewidth=2.) for i in history_dict.keys() if i!='lr']
        plt.legend()

        filename_plot = path_out_folder + 'history_dict_training_' + out_name + '.pdf'
        plt.savefig(filename_plot)
        log.info('Saved plot in: ' + filename_plot)
        plt.close()

        # Save history details
        for j_key in history_dict.keys():           # Convert to string every number to make it JSON serializable
            history_dict[j_key] = list(str(i) for i in history_dict[j_key])
        with open(json_history_filename + out_name + '.json', 'w') as j_f:
            json.dump(history_dict, j_f, indent=4, ensure_ascii=False)

        return

    ############### Warming-up ###############
    print('\n\n **** Warming-up ****\n\n')
    # Keep a copy of the weights of layer1 for later reference
    initial_layer1_weights_values = model.layers[1].get_weights()

    # Freeze the model layers up to 'last_layer' (excluded) and Recompile the model
    for i_layer in model.layers:
        i_layer.trainable = False
        if i_layer.name == layer_names_to_update[-1]:     # stop when reach the 'last_layer' point
            break
    model.compile(loss=model.loss, optimizer=model.optimizer, metrics=list(Loss_functions.values()))
    
    # Train the missing layers first to warm up the model 
    history = train_my_net(model, fine_tuning=False, final_model_name='model_trained.h5')
    plot_and_save_history(history, out_name='warming_up')

    # Check that the weights of layer1 have not changed during training
    final_layer1_weights_values = model.layers[1].get_weights()
    np.testing.assert_allclose(initial_layer1_weights_values[0], final_layer1_weights_values[0])
    np.testing.assert_allclose(initial_layer1_weights_values[1], final_layer1_weights_values[1])

    ############### Fine tuning ###############
    print('\n\n **** Fine tuning ****\n\n')
    # Un-Freeze the entire model
    for i_layer in model.layers:
        i_layer.trainable = True   
    adamopt = keras.optimizers.Adam(lr=lr_fineTuning, beta_1=0.9, beta_2=0.999) 
    model.compile(loss=model.loss, optimizer=adamopt, metrics=list(Loss_functions.values()))

    # Perform the fine-tuning lowering the 'lr' and 
    history = train_my_net(model, fine_tuning=True, final_model_name='model_trained_fine_tuning.h5')
    plot_and_save_history(history, out_name='fine_tuning')
    

    ## Testing: Plot and save results
    # Load testing data
    X_test, Y_test = volume_manager.loadDatasetFromListFiles(X_test_fullpath, y_test, voxelwise_mean, voxelwise_std, data_dims)
    log.info('Testing set shape: ' + str(X_test.shape))
    log.info('Testing set shape (GT): ' + str(Y_test.shape))
    experiment_dict["data"].update({"X_test.shape" : X_test.shape})
    
    # Testing (numerically)
    log.info('*** Testing... ***')

    train_test_fun.testModel(model, X_val, Y_val, 'Validation set')
    train_test_fun.testModel(model, X_test, Y_test, 'Testing set')
    
    # Test saving out volumes: Export volume and probability maps
    log.info('Test and save volumes')
    train_test_fun.testModelAndSaveOutVolumes(model, X_test, Y_test, X_test_fullpath, path_out_folder)

    # Save experiment details
    with open(json_filename, 'w') as j_f:
        json.dump(experiment_dict, j_f, indent=4, ensure_ascii=False)
        
    return


if __name__ == "__main__": 

    # Check if there are enough resources to train the model
#    if python_utils.checkGPUsAvailability(n_gpus=4) != True:
#        sys.exit(0)
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Classifier training')
    
    parser.add_argument('--training_name', type=str, help='Starting model')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate (float)')
    parser.add_argument('--lr_fineTuning', type=float, default=0.00001, help='learning rate (float)')
    parser.add_argument('--epochs', type=int, default=110, help='epochs (int)')
    parser.add_argument('--dropout', action='store_true', help='use dropout (add flag to use it)')
    parser.add_argument('--weight_class', action='store_true', help='weight the class numerosity (add flag to use it)')
    parser.add_argument('--loss_funct', default='NaN', choices=list(Loss_functions.keys()), help='loss function')
    parser.add_argument('--last_layer', type=int, default=1, choices=[i for i in range(1,15)],
                                        help='last layer to freeze: goes from 1 (change only output) \
                                                to 14 (change everything) (int)')
    
    parser.add_argument('--anat_ide', default='T1w_cut', type=str, help='T1w identifier (str)')
    parser.add_argument('--GT_to_predict', default='FS7_aseg_7classes', type=str, help='GT identifier (str)')

    args = parser.parse_args()
        
    # Model args
    training_name = args.training_name
    lr = float(args.learning_rate)
    lr_fineTuning = float(args.lr_fineTuning)
    epochs = args.epochs
    dropout = args.dropout
    weight_class = args.weight_class
    my_loss_funct = args.loss_funct
    last_layer = args.last_layer

    # Data args    
    anat_ide = args.anat_ide + '.nii.gz'
    GT_to_predict = args.GT_to_predict + '.nii.gz'
    batch_size = 1
        
    # Paths of mean/std volumes
    full_path_mean_subject = opj(Path_in_meanStd, '_' + args.anat_ide + '_average_subject.nii.gz')
    full_path_std_subject = opj(Path_in_meanStd, '_' + args.anat_ide + '_stdDev_subject.nii.gz')
        
    # Initialisation
    exper_description = 'anatIde=' + args.anat_ide.replace('_','-') + '_GT=' + \
                        (args.GT_to_predict).replace('_','-') + \
                        '_epoch=' + str(epochs)

    path_out_folder = python_utils.initialiseOutputFolder(Path_out_dir,
                                                          log_description=exper_description,
                                                          initial_desciption='fineTuning_')     # Create out folder
    print('folder: ' + str(path_out_folder.split('/')[-2]))
    Log_filename = path_out_folder + 'training.log'    
    log = python_utils.initialiseLogger(Log_filename, log_level=20)     # 20: INFO
    python_utils.logEveryPackageLoad(log)


    # Save a json file with every useful info about the training (paths, model, etc.)
    json_filename = path_out_folder + 'experiment_detail.json'
    json_history_filename = path_out_folder + 'training_history_'
    experiment_dict = {}
    experiment_dict['original_experiment'] = opj(Path_in_models, training_name, 'experiment_detail.json')
    experiment_dict['experiment_name'] = path_out_folder.split('/')[-2]
    experiment_dict['path'] = {"path_out_folder" : path_out_folder,
                                "Log_filename" : Log_filename,
                                "Path_in_data" : Path_in_data,
                                "full_path_mean_subject" : full_path_mean_subject,
                                "full_path_std_subject" : full_path_std_subject
                                }
    experiment_dict['model'] = {"lr" : lr,
                                "epochs" : epochs,
                                "dropout" : dropout,
                                "weight_class" : weight_class,
                                "last_layer" : last_layer,
                                }
    experiment_dict['data'] = {"anat_ide" : anat_ide,
                               "GT_to_predict" : GT_to_predict,
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

    main(log, my_loss_funct)







# # Model args: testing parameters
# lr = 0.00001
# epochs = 1
# batch_size = 1
# augmentation = False
# last_layer = 1
# dropout = True
# weight_class = True
# # Data args    
# anat_ide = 'T1w_cut.nii.gz'
# GT_to_predict = 'FS7_aseg_7classes.nii.gz'

# training_name = Training_name

# full_path_mean_subject = opj(Path_in_meanStd, '_T1w_cut_average_subject.nii.gz')
# full_path_std_subject = opj(Path_in_meanStd, '_T1w_cut_stdDev_subject.nii.gz')

# exper_description = 'anatIde=' + 'T1w_cut'.replace('_','-') + '_GT=' + ('FS7_aseg+aparc_7classes').replace('_','-') + '_epoch=' + str(epochs)









