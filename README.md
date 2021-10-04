# <p align="center">CEREBRUM-7T</p>


Visit the [Project page](https://rocknroll87q.github.io/cerebrum7t/).

## Description

Implementation of the paper "CEREBRUM 7T: Fast and Fully-volumetric Brain Segmentation of 7 Tesla MR Volumes" ([link](https://doi.org/10.1002/hbm.25636)).

<p align="center">
<img src="https://github.com/rockNroll87q/cerebrum7t/blob/master/misc/graphical_abstract.png" width="700" />  
</p>

In the paper, we tackle the problem of automatic 7T MRI segmentation. 
The generated model is able to produce accurate multi-structure segmentation masks on six different classes, in only few seconds.
Classes are: gray matter (GM), white matter (WM), cerebrospinal fluid (CSF), ventricles, cerebellum, brainstem, and basal ganglia.

## Usage

Visit the relative [page](https://rocknroll87q.github.io/cerebrum7t/usage) to learn how to use `CEREBRUM-7T` from source code, docker, or singularity.

## Data

Visit the relative [page](https://rocknroll87q.github.io/cerebrum7t/data) for all the information needed about the data.

## Authors

[Michele Svanera](https://github.com/rockNroll87q)
&
[Dennis Bontempi](https://github.com/denbonte)


## Citation

If you find this code useful in your research, please consider citing our paper:

```
@article{SvaneraHBM21Cerebrum7T,
	author = {Svanera, Michele and Benini, Sergio and Bontempi, Dennis and Muckli, Lars},
	title = {CEREBRUM-7T: Fast and Fully Volumetric Brain Segmentation of 7 Tesla MR Volumes},
	journal = {Human Brain Mapping},
	volume = {n/a},
	number = {n/a},
	pages = {},
	keywords = {3D image analysis, brain MRI segmentation, convolutional neural networks, weakly supervised learning},
	doi = {https://doi.org/10.1002/hbm.25636},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.25636},
	eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.25636},
}
```
