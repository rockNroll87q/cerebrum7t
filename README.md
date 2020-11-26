# <p align="center">CEREBRUM-7T</p>


Visit the [Project page](https://rocknroll87q.github.io/cerebrum7t/).

## Description

Implementation of the paper "CEREBRUM-7T: fast and fully-volumetric  brain segmentation of 7 Tesla MR volumes" ([link](https://www.biorxiv.org/content/10.1101/2020.07.07.191536v1.full)).

<p align="center">
<img src="https://github.com/rockNroll87q/cerebrum7t/blob/master/misc/graphical_abstract.png" width="700" />  
</p>

In the paper, we tackle the problem of automatic 7T MRI segmentation. 
The generated model is able to produce accurate multi-structure segmentation masks on six different classes, in only few seconds.
Classes are: gray matter (GM), white matter (WM), cerebrospinal fluid (CSF), ventricles, cerebellum, brainstem, and basal ganglia.

## Usage

## Data

At this link you can download the data; you need an EBRAINS account, that you can get [here](https://ebrains.eu/register/). 

### Requirements

The code needs some dependencies that you can find in `requirements.txt`. Download the repo and run:

```
pip install -r requirements.txt
```

to install all the dependencies. 
<i><b>N.B.</b> `tensorflow-gpu` not needed for testing, but strongly suggested for training.</i>

## Authors

[Michele Svanera](https://github.com/rockNroll87q)
&
[Dennis Bontempi](https://github.com/denbonte)


## Citation

If you find this code useful in your research, please consider citing our paper:

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
