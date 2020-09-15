# Data Directory `data/`


## Directory Structure

In our work, we used out-of-the-scanner data. The only pre-process the data went through was the conversion from DICOM to the NIfTI format, carried out using [dcm2niix](https://github.com/rordenlab/dcm2niix).

For the following reason, in order to run the code found in this repo "off-the-shelf", data must stored in NIfTI files (`.nii` or `.nii.gz`). We used [Nipy's NiBabel](https://nipy.org/nibabel/) to handle such MR Images.

Applying changes in these regards (e.g., to use other file formats, or handle data exploiting other python libraries) it's pretty straight forward, as the aforementioned library is used for loading and saving operations.

The data in this folder should be organised as follows:

```
data/
├── sub-001
│   ├── anat
│   │   ├── sub-001_ses-001_INV1.json
│   │   ├── sub-001_ses-001_INV1.nii.gz
│   │   ├── sub-001_ses-001_INV2.json
│   │   ├── sub-001_ses-001_INV2.nii.gz
│   │   ├── sub-001_ses-001_T1w.json
│   │   ├── sub-001_ses-001_T1w.nii.gz
│   └── seg
│       ├── sub-001_ses-001_Fracasso16-mc.nii.gz
│       ├── sub-001_ses-001_Fracasso16.nii.gz
│       ├── sub-001_ses-001_FS6.nii.gz
│       ├── sub-001_ses-001_FS7.nii.gz
├── sub-002
│   ├── anat
│   │   ├── sub-002_ses-001_INV1.json
│   │   ├── sub-002_ses-001_INV1.nii.gz
│   │   ├── sub-002_ses-001_INV2.json
│   │   ├── sub-002_ses-001_INV2.nii.gz
│   │   ├── sub-002_ses-001_T1w.json
│   │   └── sub-002_ses-001_T1w.nii.gz
│   └── seg
│       ├── sub-002_ses-001_Fracasso16-mc.nii.gz
│       ├── sub-002_ses-001_Fracasso16.nii.gz
│       ├── sub-002_ses-001_FS6.nii.gz
│       └── sub-002_ses-001_FS7.nii.gz
├── sub-003
│   ├── anat
│   │   ├── sub-003_ses-001_INV1.json
│   │   ├── sub-003_ses-001_INV1.nii.gz

...
 
```


## Publicly Available Test Data

Along with the publicly available data, an 6-class segmentation mask is provided. The segmented classes (and the color code used in the notebooks and in the paper) are:

| Class ID | Substructure/Tissue |    Color    |
|:--------:|:-------------------:|:-----------:|
|     0    |      Background     | Transparent |
|     1    |     Grey matter     | Light green |
|     2    |    Basal ganglia    |  Dark green |
|     3    |     White matter    |     Red     |
|     4    |      Ventricles     |     Blue    |
|     5    |      Cerebellum     |    Yellow   |
|     6    |      Brainstem      |     Pink    |


Such ground truth was obtained starting from FreeSurfer's `recon-all` procedure, merging the classes such that only the ones [used in the MICCAI MRBrainS13 and MRBrainS18 challenges](https://mrbrains13.isi.uu.nl/data/) were kept (except the less numerous classes - "White matter lesions", "Infarction" and "Other" - not directly obtainable from the atlas-based segmentation, or for which we considered the latter to be too little reliable).
