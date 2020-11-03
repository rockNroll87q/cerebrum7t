---
layout: page
title: <a href="https://rocknroll87q.github.io/cerebrum7t/" target="_blank">CEREBRUM 7T</a>
---

[<-- main page](https://rocknroll87q.github.io/cerebrum7t/)

<hr>
# AHEAD: a fine tuning experiment

In this experiment, we accomplish very good results using only 20 volumes to `fine tuning` the model (trained on Glasgow data) on AHEAD data  ([link](https://doi.org/10.1016/j.neuroimage.2020.117200)).
The labels used for training were those described in the paper, while the labels for fine tunings derive from FreeSurfer v7.
Comparisons that you can see below are made against FreeSurfer v7 on the testing set.

As it is possible to see, FreeSurfer v7, which has been improved for UHF data, is able to segment very well multiple areas (ex. GM/WM boundary), but the inhomogeneity of the scan affects its ability to correctly select all the areas (ex. parietal and temporal lobes). It produces some "holes" in the segmentations.
In addition, our method produces much smoother results.

For further inspections, yon can download the segmentation masks for both methods [here](https://github.com/rockNroll87q/cerebrum7t/tree/gh-pages/results/AHEAD/seg_labels).
<!--  <tr>
    <td><b style="font-size:30px"></b>sub_0068</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0068_FreeSurfer.gif" width="256" height="352"/></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0068_CEREBRUM.gif" width="256" height="352" />  </center></td>
 </tr>-->
The segmented classes and the color code used below are:

| Class ID | Substructure/Tissue |    Color    |
|:--------:|:-------------------:|:-----------:|
|     0    |      Background     | Transparent |
|     1    |     Grey matter     | Light green |
|     2    |    Basal ganglia    |  Dark green |
|     3    |     White matter    |     Red     |
|     4    |      Ventricles     |     Blue    |
|     5    |      Cerebellum     |    Yellow   |
|     6    |      Brainstem      |     Pink    |

### Results

<table align="center" cellspacing="0" cellpadding="0">
 <tr>
 	<td><b style="font-size:20px">Subj ID</b></td>
 	<td><center><b style="font-size:20px">FreeSurfer v7</b></center></td> 
 	<td><center><b style="font-size:20px">CEREBRUM 7T</b></center></td> 
 </tr>
 

 <tr>
    <td><b style="font-size:30px"></b>sub_0014 (WM only)</td>
    <td><center><img src="./results/AHEAD/meshes/sub-0014_FreeSurfer.gif" width="400" /></center></td>
    <td><center><img src="./results/AHEAD/meshes/sub-0014_CEREBRUM.gif" width="400" />  </center></td>
 </tr>
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0000</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0000_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0000_CEREBRUM.gif"  />  </center></td>
 </tr>
 
 <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

 <tr>
    <td><b style="font-size:30px"></b>sub_0014</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0014_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0014_CEREBRUM.gif"  />  </center></td>
 </tr>
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0025</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0025_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0025_CEREBRUM.gif"  />  </center></td>
 </tr>
 
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0035</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0035_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0035_CEREBRUM.gif"  />  </center></td>
 </tr>
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0011</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0011_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0011_CEREBRUM.gif"  />  </center></td>
 </tr>

 <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>


  <tr>
    <td><b style="font-size:30px"></b>sub_0037</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0037_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0037_CEREBRUM.gif"  />  </center></td>
 </tr>
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0038</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0038_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0038_CEREBRUM.gif"  />  </center></td>
 </tr>
 
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>


  <tr>
    <td><b style="font-size:30px"></b>sub_0040</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0040_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0040_CEREBRUM.gif"  />  </center></td>
 </tr>
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0068</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0068_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0068_CEREBRUM.gif"  />  </center></td>
 </tr>
  <tr>
 	<td style="font-size:15px">Subj ID</td>
 	<td><center style="font-size:15px">FreeSurfer v7</center></td> 
 	<td><center style="font-size:15px">CEREBRUM 7T</center></td> 
 </tr>

  <tr>
    <td><b style="font-size:30px"></b>sub_0093</td>
    <td><center><img src="./results/AHEAD/GIF/sub-0093_FreeSurfer.gif" /></center></td>
    <td><center><img src="./results/AHEAD/GIF/sub-0093_CEREBRUM.gif"  />  </center></td>
 </tr>
</table>
