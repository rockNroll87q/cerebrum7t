---
layout: page
title: <a href="https://rocknroll87q.github.io/cerebrum7t/">CEREBRUM 7T</a>
---

[<-- main page](https://rocknroll87q.github.io/cerebrum7t/)

<hr>
# From src: virtual environment

Please note that these instructions don't require you to be a super user. Everything can be done in user mode, i.e., you can setup this in a server. 
The only requirement is the you need the right version of CUDA (10.0) that match TensorFlow 1.14.

### Setup

To create a virtual environment, go to your project’s directory and run venv:

`python3 -m venv cerebrum7T`

Activate the virtual environment:

`source cerebrum7T/bin/activate`

Upgrade pip:

`python3 -m pip install --upgrade pip`

Install all the dependencies with (it takes a while):

`python3 -m pip install -r requirements.txt`

Enjoy!

### Training

~~~
usage: training.py [-h] [--model {Input,ThreeLevelsConvUNetStridedConv}]
                   [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                   [--epochs EPOCHS] [--n_filters N_FILTERS]
                   [--loss_funct {categorical_crossentropy,dice_coef_multilabel,dice_coef_multilabel_metric,tversky_loss,jaccard_metric,dice_coef_metric,dice_coef_loss}]
                   [--encoder_act_funct {relu,elu}]
                   [--decoder_act_funct {relu,elu}] [--dropout]
                   [--weight_class] [--anat_ide ANAT_IDE]
                   [--GT_to_predict GT_TO_PREDICT] [--augmentation]
                   [--augm_factor AUGM_FACTOR]

Classifier training

optional arguments:
  -h, --help            show this help message and exit
  --model {Input,ThreeLevelsConvUNetStridedConv}
                        Model name
  --learning_rate LEARNING_RATE
                        learning rate (float)
  --batch_size BATCH_SIZE
                        batch_size (int)
  --epochs EPOCHS       epochs (int)
  --n_filters N_FILTERS
                        number of filters for the first layer (double each
                        layer)
  --loss_funct {categorical_crossentropy,dice_coef_multilabel,dice_coef_multilabel_metric,tversky_loss,jaccard_metric,dice_coef_metric,dice_coef_loss}
                        loss function
  --encoder_act_funct {relu,elu}
                        Activation function encoder.
  --decoder_act_funct {relu,elu}
                        Activation function encoder.
  --dropout             use dropout (add flag to use it)
  --weight_class        weight the class numerosity (add flag to use it)
  --anat_ide ANAT_IDE   T1w identifier (str)
  --GT_to_predict GT_TO_PREDICT
                        GT identifier (str)
  --augmentation        use augmentation (add flag to use it)
  --augm_factor AUGM_FACTOR
                        factor of: how many times the training set use for
                        augmentation
~~~
### Fine-tuning

~~~
usage: fine_tuning.py [-h] [--training_name TRAINING_NAME]
                      [--learning_rate LEARNING_RATE]
                      [--lr_fineTuning LR_FINETUNING] [--epochs EPOCHS]
                      [--dropout] [--weight_class]
                      [--loss_funct {categorical_crossentropy,dice_coef_multilabel,dice_coef_multilabel_metric,tversky_loss,jaccard_metric,dice_coef_metric,dice_coef_loss}]
                      [--last_layer {1,2,3,4,5,6,7,8,9,10,11,12,13,14}]
                      [--anat_ide ANAT_IDE] [--GT_to_predict GT_TO_PREDICT]

Classifier training

optional arguments:
  -h, --help            show this help message and exit
  --training_name TRAINING_NAME
                        Starting model
  --learning_rate LEARNING_RATE
                        learning rate (float)
  --lr_fineTuning LR_FINETUNING
                        learning rate (float)
  --epochs EPOCHS       epochs (int)
  --dropout             use dropout (add flag to use it)
  --weight_class        weight the class numerosity (add flag to use it)
  --loss_funct {categorical_crossentropy,dice_coef_multilabel,dice_coef_multilabel_metric,tversky_loss,jaccard_metric,dice_coef_metric,dice_coef_loss}
                        loss function
  --last_layer {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
                        last layer to freeze: goes from 1 (change only output)
                        to 14 (change everything) (int)
  --anat_ide ANAT_IDE   T1w identifier (str)
  --GT_to_predict GT_TO_PREDICT
                        GT identifier (str)
~~~

### Testing
~~~
usage: testing.py [-h] --training_name TRAINING_NAME

Classifier testing on BIDS

optional arguments:
  -h, --help            show this help message and exit
  --training_name TRAINING_NAME
                        Starting model
~~~

<hr>
# Docker

These commands assumes that the data is organised as described here ([link](https://github.com/rockNroll87q/cerebrum7t/blob/master/data/README.md)).

Download the last version of `cerebrum7t` with `docker pull rocknroll87q/cerebrum7t` or download the `.tar` from [here](https://cloud.psy.gla.ac.uk/index.php/s/CNZDdvvnjJ4iBx3) (psw: `rocknroll87q/cerebrum7t`).

### Training

~~~
docker run -it \
-v /path/to/BIDS/:/cerebrum7t/data/BIDS/ \
-v /path/to/BIDS_augm/:/cerebrum7t/data/BIDS_augm/ \
-v /path/to/output/:/cerebrum7t/output/ \
-v /path/to/inhomogeneity_volume.npy:/cerebrum7t/data/inhomogeneity_volume.npy \
-v /path/to/mean_std/:/cerebrum7t/data/mean_std/ \
cerebrum7t \
python /cerebrum7t/src/training.py --learning_rate 0.0005 --GT_to_predict 'training_labels' --anat_ide 'T1w' --augmentation --augm_factor 10 --dropout
~~~

### Fine-tuning

~~~
docker run -it \
-v /path/to/BIDS/:/cerebrum7t/data/BIDS/ \
-v /path/to/BIDS_augm/:/cerebrum7t/data/BIDS_augm/ \
-v /path/to/output/:/cerebrum7t/output/fine_tuning/ \
-v /path/to/training/:/cerebrum7t/output/training/ \
-v /path/to/inhomogeneity_volume.npy:/cerebrum7t/data/inhomogeneity_volume.npy \
-v /path/to/training_samples.csv:/cerebrum7t/data/training_samples.csv \
-v /path/to/mean_std/:/cerebrum7t/data/mean_std/ \
cerebrum7t \
python /cerebrum7t/src/fine_tuning.py --learning_rate 0.0001 --GT_to_predict 'FreeSurfer_v7' --anat_ide 'T1w' --weight_class --dropout --last_layer 3 --loss_funct 'dice_coef_multilabel' --training_name 'training_YYYY-MM-DD_etc'
~~~

### Testing

~~~
docker run -it \
-v /path/to/training/:/cerebrum7t/output/training/ \
-v /path/to/mean_std/:/cerebrum7t/data/mean_std/ \
-v /path/to/testing_volume.nii.gz \
-v /path/to/output/:/cerebrum7t/output/testing/ \
cerebrum7t \
python /cerebrum7t/src/testing.py --training_name 'training_YYYY-MM-DD_etc'
~~~


<hr>
# Singularity

These commands assumes that the data is organised as described here ([link](https://github.com/rockNroll87q/cerebrum7t/blob/master/data/README.md)). If your data has a different structure, you just need to update the code.

Download the last version of `cerebrum7t` at [this link](https://cloud.psy.gla.ac.uk/index.php/s/OUeuBjoxavTJq1Z) (psw: `rocknroll87q/cerebrum7t`).

### Training

~~~
export SINGULARITY_HOME=$PWD:/home/$USER
cd /

singularity exec --nv \
-B /path/to/BIDS/:/cerebrum7t/data/BIDS/ \
-B /path/to/BIDS_augm/:/cerebrum7t/data/BIDS_augm/ \
-B /path/to/output/:/cerebrum7t/output/ \
-B /path/to/inhomogeneity_volume.npy:/cerebrum7t/data/inhomogeneity_volume.npy \
-B /path/to/mean_std/:/cerebrum7t/data/mean_std/ \
/path/to/cerebrum7t_singularity.simg \
python /cerebrum7t/src/training.py --learning_rate 0.0005 --GT_to_predict 'training_labels' --anat_ide 'T1w' --augmentation --augm_factor 10 --dropout
~~~




### Fine-tuning


~~~
export SINGULARITY_HOME=$PWD:/home/$USER
cd /

singularity exec --nv \
-B /path/to/BIDS/:/cerebrum7t/data/BIDS/ \
-B /path/to/BIDS_augm/:/cerebrum7t/data/BIDS_augm/ \
-B /path/to/output/:/cerebrum7t/output/fine_tuning/ \
-B /path/to/training/:/cerebrum7t/output/training/ \
-B /path/to/inhomogeneity_volume.npy:/cerebrum7t/data/inhomogeneity_volume.npy \
-B /path/to/training_samples.csv:/cerebrum7t/data/training_samples.csv \
-B /path/to/mean_std/:/cerebrum7t/data/mean_std/ \
/path/to/cerebrum7t_singularity.simg \
python /cerebrum7t/src/fine_tuning.py --learning_rate 0.0001 --GT_to_predict 'FreeSurfer_v7' --anat_ide 'T1w' --weight_class --dropout --last_layer 3 --loss_funct 'dice_coef_multilabel' --training_name 'training_YYYY-MM-DD_etc'
~~~


 
### Testing

~~~
export SINGULARITY_HOME=$PWD:/home/$USER
cd /

singularity exec --nv \
-B /path/to/training/:/cerebrum7t/output/training/ \
-B /path/to/mean_std/:/cerebrum7t/data/mean_std/ \
-B /path/to/testing_volume.nii.gz \
-B /path/to/output/:/cerebrum7t/output/testing/ \
/path/to/cerebrum7t_singularity.simg \
python /cerebrum7t/src/testing.py --training_name 'training_YYYY-MM-DD_etc'
~~~

