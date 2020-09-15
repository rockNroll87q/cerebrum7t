# <p align="center">CEREBRUM-7T</p>

## Description

Python implementation of the paper "CEREBRUM-7T: fast and fully-volumetric brain segmentation of out-of-the-scanner 7T MR volumes" ([link](https://www.biorxiv.org/content/10.1101/2020.07.07.191536v1.full))

In the paper, we tackle the problem of automatic 7T MRI segmentation. 
The generated model is able to produce accurate multi-structure segmentation masks on six different classes, in only few seconds.
Classes are: gray matter (GM), white matter (WM), cerebrospinal fluid (CSF), ventricles, cerebellum, brainstem, and basal ganglia.

## Usage

### Requirements

The code needs some dependencies that you can find in `requirements.txt`. Download the repo and run:

```
pip install -r requirements.txt
```

to install all the dependencies. 
<i><b>N.B.</b> `tensorflow-gpu` is not needed neither for the training nor for the testing, but performance greatly degrades if the use of GPUs is not supported.</i>


## Citation

If you find this code useful in your research, please, consider citing our paper:

```
@article {Svanera2020.07.07.191536,
	author = {Svanera, Michele and Bontempi, Dennis and Benini, Sergio and Muckli, Lars},
	title = {CEREBRUM-7T: fast and fully-volumetric brain segmentation of out-of-the-scanner 7T MR volumes},
	elocation-id = {2020.07.07.191536},
	year = {2020},
	doi = {10.1101/2020.07.07.191536},
	URL = {https://www.biorxiv.org/content/early/2020/07/08/2020.07.07.191536},
	eprint = {https://www.biorxiv.org/content/early/2020/07/08/2020.07.07.191536.full.pdf},
	journal = {bioRxiv}
}
```
