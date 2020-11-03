---
layout: page
title: CEREBRUM 7T
---

[<-- main page](https://rocknroll87q.github.io/cerebrum7t/)

<hr>
# Schneider et al. 19: a fine tuning experiment on manual segmentation


In this experiment, we tried to push the limit of our method in this very challenging scenario. 
Using the dataset below, we fine tune the model trained on Glasgow data to replicate the manual segmentations provided.

~~~
Marian Schneider, Faruk Omer Gulban, & Rainer Goebel. (2019).
Data set for sub-millimetre MRI tissue class segmentation (Version v1.0.0) 
[Data set]. Zenodo.
http://doi.org/10.5281/zenodo.3401388
~~~

The dataset has excellent manual segmentations, but only for 4 volumes of the `mp2rage` sequence.
We exploit cross-validation using 3 volumes for training and 1 for testing.
Since the method needs a substantial training set, we applied a strong augmentation procedure, concatenating different strategies of volume manipulation: like translation, rotation, and morphing. 
We then created 30 volumes for each training sample (for a total of 90). 
Due to the GPU memory limitations, we had to suppress the `sagittal sinus` label.
To easier the task of our method, we applied a brain mask to the volume, cropping at outside of the the skull.

Results show that, with only 3 volumes for fine-tuning, if the predicted labels are very accurate, and the model provide great flexibility.
To analyse these results, we need to distinguish between two different scenarios. 
In the first, we have the same classes as before, like GM, WM, and ventricles, or a combination of previous classes, like CSF or subcortical, which is a combination of basal ganglia, brainstem, and cerebellum.
On these classes, the model takes advantage of the previous learning (on Glasgow data) and simply transfer/applies the knowledge on the new dataset.
In the second scenario, on totally new classes, like vessels, the model has not a prior-knowledge and the segmentation results are lower.
This was expected and results confirmed our hypothesis.
Also here our method produces smooth segmentation masks.

In general, these results are very important because show that it is possible to deal with one of the main limitation of deep learning: the need for a big dataset.
However, it is pretty straightforward to say that in this scenario, more affidabile strategies can be applied, like decomposing the volumes in slices and apply a slice-based method. 

For further inspections, yon can download the segmentation masks for both manual and our method [here](https://github.com/rockNroll87q/cerebrum7t/tree/gh-pages/results/Schneider19/seg_labels).


### Results

<table align="center" cellspacing="0" cellpadding="0">
 <tr>
 	<td><b style="font-size:20px">Subj ID</b></td>
 	<td><center><b style="font-size:20px">Manual</b></center></td> 
 	<td><center><b style="font-size:20px">CEREBRUM 7T</b></center></td> 
 </tr>
 
   <tr>
    <td><b style="font-size:30px"></b>sub-019 (WM only)</td>
    <td><center><img src="./results/Schneider19/meshes/sub-019_uniCut_defaced_gt_white_matter.gif" /></center></td>
    <td><center><img src="./results/Schneider19/meshes/sub-019_uniCut_defaced_predicted_volume_white_matter.gif"  />  </center></td>
 </tr>
 

 <tr>
    <td><b style="font-size:30px"></b>sub-019 (GM only)</td>
    <td><center><img src="./results/Schneider19/meshes/sub-019_uniCut_defaced_gt_grey_matter.gif" width="400" /></center></td>
    <td><center><img src="./results/Schneider19/meshes/sub-019_uniCut_defaced_predicted_volume_grey_matter.gif" width="400" />  </center></td>
 </tr>

 <tr>
    <td><b style="font-size:30px"></b>sub-019 (vessels only)</td>
    <td><center><img src="./results/Schneider19/meshes/sub-019_uniCut_defaced_gt_vessels.gif" width="400" /></center></td>
    <td><center><img src="./results/Schneider19/meshes/sub-019_uniCut_defaced_predicted_volume_vessels.gif" width="400" />  </center></td>
 </tr>
 
 </table>
