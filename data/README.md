# Data Directory `data/`


## Directory Structure

In our work, we used plain reconstructed data. 
The only pre-process applied to the data is the conversion from DICOM to the NIfTI format, carried out using [dcm2niix](https://github.com/rordenlab/dcm2niix).
All the volumes are in NIfTI format (`.nii` or `.nii.gz`). 
We used [Nipy's NiBabel](https://nipy.org/nibabel/) to handle such MR Images.

The data should be organised as follows, for the subject N session S:

```
BIDS/
├── sub-00N
│   ├── anat
│   │   ├── sub-00N_ses-00S_INV1.json			  -> INV1 descriptors
│   │   ├── sub-00N_ses-00S_INV1.nii.gz			-> INV1 volume
│   │   ├── sub-00N_ses-00S_INV2.json			  -> INV2 descriptors
│   │   ├── sub-00N_ses-00S_INV2.nii.gz			-> INV1 volume
│   │   ├── sub-00N_ses-00S_T1w.json			   -> MP2RAGE descriptors
│   │   ├── sub-00N_ses-00S_T1w.nii.gz			 -> MP2RAGE volume
│   └── seg
│       ├── sub-00N_ses-00S_CEREBRUM7T.nii.gz		    -> segmentation by our method (only for testing volumes)
│       ├── sub-00N_ses-00S_Fracasso16.nii.gz		    -> segmentation by Fracasso (2016)
│       ├── sub-00N_ses-00S_FreeSurfer_v6.nii.gz		 -> segmentation by FreeSurfer v06
│       ├── sub-00N_ses-00S_FreeSurfer_v7.nii.gz		 -> segmentation by FreeSurfer v07
│       ├── sub-00N_ses-00S_nighres.nii.gz		 	     -> segmentation by Huntenburg (2018)
│       ├── sub-00N_ses-00S_training_labels.nii.gz	-> segmentation mask used for training 
```

Which is the same structure you can find in the EBRAINS release.
Using a different data structure is possible, but the code needs to be slightly modified.


## Tissue classes

Along with the publicly available data, 6-class (+ background) segmentation masks are provided. The segmented classes (and the color code used in the notebooks and in the paper) are:

| Class ID | Substructure/Tissue |    Color    |
|:--------:|:-------------------:|:-----------:|
|     0    |      Background     | Transparent |
|     1    |     Grey matter     | Light green |
|     2    |    Basal ganglia    |  Dark green |
|     3    |     White matter    |     Red     |
|     4    |      Ventricles     |     Blue    |
|     5    |      Cerebellum     |    Yellow   |
|     6    |      Brainstem      |     Pink    |


Such ground truth was obtained using ad-hoc procedure (see the manuscript) using the classes [used in the MICCAI MRBrainS13 and MRBrainS18 challenges](https://mrbrains13.isi.uu.nl/data/).
