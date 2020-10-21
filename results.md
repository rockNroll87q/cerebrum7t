---
layout: page
title: CEREBRUM 7T
---

[<-- main page](https://rocknroll87q.github.io/cerebrum7t/)

# Results

In this page, there are results of our method with data from two different sites, both T1w: on Glasgow data and on AHEAD data.

## Glasgow




## AHEAD: a `fine tuning` experiment

In this experiment, we accomplish very good results using only 20 volumes to `fine tuning` the model (trained on Glasgow data) on AHEAD data  ([link](https://doi.org/10.1016/j.neuroimage.2020.117200)).
The labels used for training were described in the paper, while the labels for fine tunings derive from FreeSurfer v7.
Comparisons that you can see below are made against FreeSurfer v7 on the testing set.

<table align="center" width="80%" cellspacing="0" cellpadding="0">
 
 <tr>
    <td><b style="font-size:30px"></b>Sub_0000</td>
    <td><center><img src="./GIF/sub-001_gt.gif" height="300"/></center></td>
    <td><center><img src="./GIF/sub-001_predicted.gif" height="300" />  </center></td>
 </tr>

 
</table>
