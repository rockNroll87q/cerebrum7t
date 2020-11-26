#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:34:44 2019

@author: Michele Svanera, Dennis Bontempi
University of Glasgow.

Functions for model definition.

"""

################################################################################################################
## Imports 

from __future__ import division, print_function

import os, shutil
import numpy as np
import logging
log = logging.getLogger(__name__)

from keras.models import Model

from keras.layers import Input, Activation, BatchNormalization, UpSampling3D
from keras.layers.merge import Add
from keras.layers.pooling import MaxPooling3D
from keras.layers.convolutional import Conv3D, Conv3DTranspose
from keras.layers.core import Dropout
import keras
import tensorflow as tf

from .Losses3D import losses_dict

from keras.callbacks import TensorBoard

################################################################################################################
## Model definitions

def ThreeLevelsConvUNetStridedConv(input_dim, n_classes, n_filters, lr, loss, 
                               encoder_act_function='elu', decoder_act_function='relu', 
                               final_activation='softmax', init='glorot_normal', use_dropout=False,
                               max_n_filters=256):
    ''' 3-levels fully-volumetric U-Net with Strided-convolution for downsampling.
    '''

    if use_dropout == True:
        lvl1_dropout = 0.0      #0.10
        lvl2_dropout = 0.10
        lvl3_dropout = 0.10
    else:
        lvl1_dropout = 0.0 
        lvl2_dropout = 0.0
        lvl3_dropout = 0.0

    lvl1_kernel_size = (3, 3, 3)
    lvl2_kernel_size = (3, 3, 3)
    lvl3_kernel_size = (3, 3, 3)


    with tf.device('/gpu:1'):
        
        # Input data
        inputs = Input(input_dim, name='main_input') 

        # -----------------------------------
        #         ENCODER - LEVEL 1
        # -----------------------------------
        
        # Level1-block1 conv layer: 1-conv (full-resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        enc_lvl1_block1_conv = Conv3D(filters = n_filters, 
                                      kernel_size = lvl1_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = encoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'enc_lvl1_block1_conv')(inputs)
        enc_lvl1_block1_dr = Dropout(rate = lvl1_dropout, 
                                     name = 'enc_lvl1_block1_dr')(enc_lvl1_block1_conv)
        
        
        # Level1 strided-conv layer: down-sampling
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        down_1to2_conv = Conv3D(filters = min(2*n_filters,max_n_filters), 
                                kernel_size = (4,4,4), 
                                strides = (4,4,4), 
                                padding = 'same', 
                                activation = encoder_act_function, 
                                kernel_initializer = init, 
                                name = 'down_1to2_conv')(enc_lvl1_block1_dr)
        down_1to2_dr = Dropout(rate = lvl1_dropout, 
                               name = 'down_1to2_dr')(down_1to2_conv)


    with tf.device('/gpu:0'):

        # -----------------------------------
        #         ENCODER - LEVEL 2
        # -----------------------------------

        # Level2-block1 conv layer: 2-conv (1/4 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        enc_lvl2_block1_conv = Conv3D(filters = min(2*n_filters,max_n_filters), 
                                      kernel_size = lvl2_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = encoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'enc_lvl2_block1_conv')(down_1to2_dr)
        enc_lvl2_block1_dr = Dropout(rate = lvl2_dropout, 
                                     name = 'enc_lvl2_block1_dr')(enc_lvl2_block1_conv)
        
        # Level2-block2 conv layer: 2-conv (1/4 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        enc_lvl2_block2_conv = Conv3D(filters = min(2*n_filters,max_n_filters), 
                                      kernel_size = lvl2_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = encoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'enc_lvl2_block2_conv')(enc_lvl2_block1_dr)
        enc_lvl2_block2_dr = Dropout(rate = lvl2_dropout, 
                                     name = 'enc_lvl2_block2_dr')(enc_lvl2_block2_conv)


        # Level2 strided-conv layer: down-sampling
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        down_2to3_conv = Conv3D(filters = min(4*n_filters,max_n_filters), 
                                kernel_size = (2,2,2), 
                                strides = (2,2,2), 
                                padding = 'same', 
                                activation = encoder_act_function, 
                                kernel_initializer = init, 
                                name = 'down_2to3_conv')(enc_lvl2_block2_dr)
        down_2to3_dr = Dropout(rate = lvl2_dropout, 
                               name = 'down_2to3_dr')(down_2to3_conv)

                                 
        # -----------------------------------
        #         BOTTLENECK LAYER (Layer3)
        # -----------------------------------
                                                
        # Level3-block1 conv layer: 3-conv (1/8 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        enc_lvl3_block1_conv = Conv3D(filters = min(4*n_filters,max_n_filters), 
                                      kernel_size = lvl3_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = encoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'enc_lvl3_block1_conv')(down_2to3_dr)
        enc_lvl3_block1_dr = Dropout(rate = lvl3_dropout, 
                                     name = 'enc_lvl3_block1_dr')(enc_lvl3_block1_conv)

        # Level3-block2 conv layer: 3-conv (1/8 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        enc_lvl3_block2_conv = Conv3D(filters = min(4*n_filters,max_n_filters), 
                                      kernel_size = lvl3_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = encoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'enc_lvl3_block2_conv')(enc_lvl3_block1_dr)
        enc_lvl3_block2_dr = Dropout(rate = lvl3_dropout, 
                                     name = 'enc_lvl3_block2_dr')(enc_lvl3_block2_conv)

    with tf.device('/gpu:3'):
        
        # Level3-block3 conv layer: 3-conv (1/8 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        enc_lvl3_block3_conv = Conv3D(filters = min(4*n_filters,max_n_filters), 
                                      kernel_size = lvl3_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = decoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'enc_lvl3_block3_conv')(enc_lvl3_block2_dr)
        enc_lvl3_block3_dr = Dropout(rate = lvl3_dropout, 
                                     name = 'enc_lvl3_block3_dr')(enc_lvl3_block3_conv)
    
    
        # Level3 strided-conv layer: up-sampling
        # ... -> Transpose Conv 3D -> Activation (None) -> BatchNorm (NO) -> Dropout -> ...
        up_3to2_conv = Conv3DTranspose(filters = min(2*n_filters,max_n_filters), 
                                       kernel_size = (2,2,2), 
                                       strides = (2,2,2), 
                                       padding = 'same', 
                                       activation = None, 
                                       kernel_initializer = init, 
                                       name = 'up_3to2_conv')(enc_lvl3_block3_dr)
        up_3to2_dr = Dropout(rate = lvl2_dropout, 
                             name = 'up_3to2_dr')(up_3to2_conv)

        
        # -----------------------------------
        #         DECODER - LEVEL 2
        # -----------------------------------

        # Level2 Skip-connection: last of second enc. level + last of third enc. level
        skip_conn_lvl2 = Add(name = 'skip_conn_lvl2')([enc_lvl2_block2_dr, up_3to2_dr])


        # Level2-block1 conv layer: 2-conv (1/4 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        dec_lvl2_block1_conv = Conv3D(filters = min(2*n_filters,max_n_filters), 
                                      kernel_size = lvl2_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = decoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'dec_lvl2_block1_conv')(skip_conn_lvl2)
        dec_lvl2_block1_dr = Dropout(rate = lvl2_dropout, 
                                     name = 'dec_lvl2_block1_dr')(dec_lvl2_block1_conv)
        
        # Level2-block2 conv layer: 2-conv (1/4 resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        dec_lvl2_block2_conv = Conv3D(filters = min(2*n_filters,max_n_filters), 
                                      kernel_size = lvl2_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = decoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'dec_lvl2_block2_conv')(dec_lvl2_block1_dr)
        dec_lvl2_block2_dr = Dropout(rate = lvl2_dropout, 
                                     name = 'dec_lvl2_block2_dr')(dec_lvl2_block2_conv)




        # Level2 strided-conv layer: up-sampling
        # ... -> Transpose Conv 3D -> Activation (None) -> BatchNorm (NO) -> Dropout -> ...
        up_2to1_conv = Conv3DTranspose(filters = n_filters, 
                                       kernel_size = (4,4,4), 
                                       strides = (4,4,4), 
                                       padding = 'same', 
                                       activation = None, 
                                       kernel_initializer = init, 
                                       name = 'up_2to1_conv')(dec_lvl2_block2_dr)
        up_2to1_dr = Dropout(rate = lvl2_dropout, 
                             name = 'up_2to1_dr')(up_2to1_conv)

        # -----------------------------------
        #         DECODER - LEVEL 1
        # -----------------------------------

        # Level1 Skip-connection: last of first enc. level + last of second dec. level
        skip_conn_lvl1 = Add(name = 'skip_conn_lvl1')([enc_lvl1_block1_dr, up_2to1_dr])

    with tf.device('/gpu:2'):

        # Level1-block1 conv layer: 1-conv (full-resolution) and dropout
        # ... -> Conv 3D -> Activation -> BatchNorm (NO) -> Dropout -> ...
        dec_lvl1_block1_conv = Conv3D(filters = n_filters, 
                                      kernel_size = lvl1_kernel_size, 
                                      strides = (1,1,1), 
                                      padding = 'same', 
                                      activation = decoder_act_function, 
                                      kernel_initializer = init, 
                                      name = 'dec_lvl1_block1_conv')(skip_conn_lvl1)
        dec_lvl1_block1_dr = Dropout(rate = lvl1_dropout, 
                                     name = 'dec_lvl1_block1_dr')(dec_lvl1_block1_conv)


        # Output data
        outputs = Conv3D(filters = n_classes, 
                         kernel_size = (1,1,1), 
                         activation = final_activation,
                         kernel_initializer = init, 
                         name = 'main_output')(dec_lvl1_block1_dr)


    # Define the model object, the optimizer, and compile them
    model = Model(inputs=[inputs], outputs=[outputs])   
    adamopt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss=losses_dict[loss], optimizer=adamopt, metrics=list(losses_dict.values()))
    
    return model
