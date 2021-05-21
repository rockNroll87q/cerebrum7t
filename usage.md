---
layout: page
title: <a href="https://rocknroll87q.github.io/cerebrum7t/">CEREBRUM 7T</a>
---

[<-- main page](https://rocknroll87q.github.io/cerebrum7t/)

<hr>
# Rationale

Few notes: the tool needs training or fine-tuning in order to be use on other data than Glasgow.

If you have enough data (~50/100 vols), a training is suggested. Otherwise fine-tuning.
For training, you can choose the shape that you want, for fine-tuning, you need to stay with the shape we used in Glasgow (256, 352, 224).

The main part of the work is to prepare the dataset: the data must have the same dimension i.e., every volume and mask have the same shapes.
There is more info here: [link](https://github.com/rockNroll87q/cerebrum7t/issues/).

In general, steps are:

1.	Prepare the dataset: collect scans and segmentation masks. Take a look [here](https://rocknroll87q.github.io/cerebrum7t/data) the structure needed for the data.
2.	Create the inhomogeneity_volume ([code](https://github.com/rockNroll87q/cerebrum7t/blob/master/src/misc/inhomogeneity_volume.py))
3.	Compute mean and std of the dataset ([code](https://github.com/rockNroll87q/cerebrum7t/blob/master/src/misc/mean_discover_BIDS.py))
4.	Data augmentation ([link](https://github.com/rockNroll87q/cerebrum7t/blob/master/src/misc/offline_data_augmentation.py)) - optional
5.	Training!



<hr>
# From source

Please note that these instructions don't require you to be a super user. Everything can be done in user mode, i.e., you can setup this in a server. 
The only requirement is the you need the right version of CUDA (10.0) that match TensorFlow 1.14.

If you don't want to use a virtual environment, simply run `pip install -r requirements.txt` to install all the dependencies. 
<b>N.B.</b> `tensorflow-gpu` is not needed for testing, but strongly suggested for training.

We also suggest to use python3.6 for this.

### Create virtual environment

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

Here the command used in the manuscript.

~~~
$ python3 ./training.py --learning_rate 0.0005 --epochs 100 --GT_to_predict 'training_labels' --anat_ide 'T1w'  --augmentation --augm_factor 10 --dropout
~~~

For an extended description of every option, please run `python3 ./training.py --help`

### Fine-tuning

Here the command used to obtained for [AHEAD results](https://rocknroll87q.github.io/cerebrum7t/results_ahead):

~~~
$ python3 ./fine_tuning.py --learning_rate 0.0001 --GT_to_predict 'FS7_aseg_7classes' --anat_ide 'T1w_cut' --weight_class --dropout --last_layer 3 --loss_funct 'dice_coef_multilabel' --lr_fineTuning 0.0005
~~~

For an extended description of every option, please run `python3 ./fine-tuning.py --help`

### Testing

~~~
$ python testing.py [-h] --training_name TRAINING_NAME
~~~

For an extended description of every option, please run `python3 ./testing.py --help`

<hr>
# Docker

These commands assumes that the data is organised as described here ([link](https://github.com/rockNroll87q/cerebrum7t/blob/master/data/README.md)).

Download the last version of `cerebrum7t` with `docker pull rocknroll87q/cerebrum7t:segmentation7T` or download the `.tar` from [here](https://cloud.psy.gla.ac.uk/index.php/s/CNZDdvvnjJ4iBx3) (psw: `rocknroll87q/cerebrum7t`).

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

