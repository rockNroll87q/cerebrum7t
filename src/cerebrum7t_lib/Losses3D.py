#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:34:44 2019

@author: Michele Svanera, Dennis Bontempi
University of Glasgow.

losses_dict = {'categorical_crossentropy'       : 'categorical_crossentropy',
               'dice_coef_multilabel'           : dice_coef_multilabel, # == tversky_loss
               'dice_coef_multilabel_metric'    : dice_coef_multilabel_metric,
               'tversky_loss'                   : tversky_loss,
               'jaccard_metric'                 : jaccard_metric,
               'dice_coef_metric'               : dice_coef_metric,
               'dice_coef_loss'                 : dice_coef_loss,
               }
"""


################################################################################################################
## Imports 

import numpy as np
import tensorflow as tf

from keras import backend as K

################################################################################################################
## Losses definitions


def tversky_loss(y_true, y_pred):
    
    # tversky coeff alpha and beta
    # if bot are set to be 0.5, the DC is obtained (exactly as from the "dice_coef_multilabel" func)
    alpha = 0.5
    beta  = 0.5
    
    # in our case, initialize a tensor of shape NxMxPxNCLASSES (M, N and P are the input dim.s)
    # NOTE: "y_true" is a NxMxPxNCLASSES tensor that contains the information about the correct
    # segmentation class (one-hot-encoded, 1.0 in that class and 0.0 in all the others)
    ones = K.ones(K.shape(y_true))
    
    # following the paper notation, name "p0" the classes prediction tensor (probability)
    p0 = y_pred
    
    # therefore, 1 - p0 is the probability that the particular voxel DOES NOT belong to that class 
    p1 = ones - y_pred
    
    # the "y_true" tensor contains 1.0 only in the channel associated to the correct class 
    # (and 0.0 in all the others) (see the example below) 
    g0 = y_true
    
    # .. so that the following quantity is a mask that can be used to keep only the probability
    # that a wrong guess is made (since the only one zeroed will be the correct one) 
    g1 = ones - y_true
    
    # .. for instance, p0[10, 10, 10, :] gives us a TF tensor (convertible to numpy array..)
    # containing the probability that the voxel at [10, 10, 10] belongs to each class
    #
    # e.g. (in case of 8 classes, GT = background):
    #       y_true[10, 10, 10, :] = g0[10, 10, 10, :] = [1.0, .0, .0, .0, .0, .0, .0, .0]
    #       y_pred[10, 10, 10, :] = p0[10, 10, 10, :] = [0.6, .1, .0, .1, .2, .0, .0, .0]
    
    # NOTE: when dealing with TF tensors, "*" in interpreted as element-wise multiplication 
    # ... so the giant sum at the numerator can be computed by element-wise multiplying the two 
    # tensors and then by computing a sum on every axis
    num = K.sum(p0*g0, (0,1,2,3))
    
    # the denominator is composed by three terms:
    #   - the numerator, so that since we're dealing with probabilities and the other two terms
    #       will always be positive, the tversky coefficient is always bounded between 0 and 1;
    #   - the tensor p0*g1 (predicted probabilities masked with g0) contains for each voxel the 
    #       probability that the classification is wrong (false positive);
    #   - the tensor p1*g0 (probability each voxel does not belong to a class masked with the
    #       tensor containing one only in the correct class) contains the probability that the 
    #       correct class is not identified (false negative);
    
    # so about the tversky coefficient:
    #   - if both the second and the third term at the denominator are small (meaning the 
    #       segmentation is quite good), then the tversky coefficient is close to one;
    #   - as alpha increases, the importance of the false positives increases;
    #   - as beta increases, the importance of the false negatives increases;
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    # compute the sum of the resulting tensor (should have dimension 1xNCLASSES)
    # this will be a number between 0 and num_classes
    ratio = K.sum(num/den) 
    
    # get the number of the classes as a tensor
    num_classes = K.cast(K.shape(y_true)[-1], 'float32')
    
    # compute the difference between the two (this will lead to something similar to the 
    # "dice_coef_multilabel", that is the sum of all the DCs associated to each class)
    return num_classes - ratio

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

def dice_coef_multilabel(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))

    #..for instance, p0[0, 10, 10, 10, :] gives us a TF tensor (convertible to numpy array..)
    # containing the probability that the voxel at [0, 10, 10, 10] belongs to each class
    #(0 is for the batch size (=1) axis)
    #
    # e.g. (in case of 7 classes, GT = background):
    #       y_true[0, 10, 10, 10, :] = g0[0, 10, 10, 10, :] = [1.0, .0, .0, .0, .0, .0, .0]
    #       y_pred[0, 10, 10, 10, :] = p0[0, 10, 10, 10, :] = [0.6, .1, .0, .1, .2, .0, .0]

    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    eps = 1e-7

    num = K.sum(p0*g0, (0 ,1 ,2 ,3)) + eps
    den = num + alpha *K.sum(p0*g1 ,(0 ,1 ,2 ,3)) + beta *K.sum(p1*g0,(0 ,1 ,2 ,3))

    ratio = K.sum(num/den)

    num_classes = K.cast(K.shape(y_true)[-1], 'float32')

    return num_classes - ratio  # it's 0 in the best case and 7 in the worst -> sum of dc on the different labels


def dice_coef_multilabel_metric(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))

    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    eps = 1e-7

    num = K.sum(p0*g0, (0 ,1 ,2 ,3)) + eps  # K.eval(num) is an array of 7 values, one for each label
    den = num + alpha *K.sum(p0*g1 ,(0 ,1 ,2 ,3)) + beta *K.sum(p1*g0,(0 ,1 ,2 ,3))

    ratio = K.sum(num/den) # sum the 7 values of num/den into 1

    return ratio/7 # to mantain the range 0 1 -> it's like the average on all the labels dc metrics


def dice_coef_multilabel_loss(y_true, y_pred):
    
    return 1 - dice_coef_multilabel_metric(y_true, y_pred)


## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

# single class dice coefficient
def dice_coef_metric(y_true, y_pred):
    
    # the smooth coefficient act as regularizer: its value altough depend from the quantity at 
    # stake (e.g. values of K.sum() ... )
    eps = 1e-7
    
    # "unwrap" the ground truth and the predicted volumes to obtain a 1xN tensors 
    # (DC doesn't take into account spatial information) (boolean vector)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #y_pred_f = K.flatten(tf.argmax(y_pred, 3))
    
    # compute the intersection between the two sets as the product between the unwrapped boolean 
    # tensors computed above (use K.sum since the quantity is still a tensor)
    intersection = K.sum(y_true_f * y_pred_f)
    
    # apply the definition of DC
    return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)

def dice_coef_loss(y_true, y_pred):
    
    return 1. - dice_coef_metric(y_true, y_pred)


def jaccard_metric(y_true, y_pred, axis=(1, 2, 3, 4)):
    """ ** from nobrainer **
    Calculate Jaccard similarity between labels and predictions.

    Jaccard similarity is in [0, 1], where 1 is perfect overlap and 0 is no
    overlap. If both labels and predictions are empty (e.g., all background),
    then Jaccard similarity is 1.

    If we assume the inputs are rank 5 [`(batch, x, y, z, classes)`], then an
    axis parameter of `(1, 2, 3)` will result in a tensor that contains a Jaccard
    score for every class in every item in the batch. The shape of this tensor
    will be `(batch, classes)`. If the inputs only have one class (e.g., binary
    segmentation), then an axis parameter of `(1, 2, 3, 4)` should be used.
    This will result in a tensor of shape `(batch,)`, where every value is the
    Jaccard similarity for that prediction.

    Implemented according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/#Equ7

    Returns
    -------
    Tensor of Jaccard similarities.

    Citations
    ---------
    Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation:
        analysis, selection, and tool. BMC Med Imaging. 2015;15:29. Published 2015
        Aug 12. doi:10.1186/s12880-015-0068-x
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    eps = tf.keras.backend.epsilon()

    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true, axis=axis) + tf.reduce_sum(y_pred, axis=axis)
    return (intersection + eps) / (union - intersection + eps)


## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------    

losses_dict = {'categorical_crossentropy'       : 'categorical_crossentropy',
               'dice_coef_multilabel'           : dice_coef_multilabel, # == tversky_loss
               'dice_coef_multilabel_metric'    : dice_coef_multilabel_metric,
               'tversky_loss'                   : tversky_loss,
               'jaccard_metric'                 : jaccard_metric,
               'dice_coef_metric'               : dice_coef_metric,
               'dice_coef_loss'                 : dice_coef_loss,
               }
    
    
    
    

