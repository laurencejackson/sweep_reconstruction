# sweep-recon

---
## Introduction
Reconstruction code for respiration resolved SWEEP acquisitions. The code presented here is a builds upon the reconstructiion pipeline described in the paper below. If you find this code useful please acknowledge this paper (in press).

>Jackson LH, Price AN, Hutter J et al. Respiration resolved imaging with continuous stable state 2D acquisition using linear frequency SWEEP. Magn Reson Med. 2019;00:1–15.

An outline of the reconstruction pipeline is given in the figure below. For more details please refer to the paper given above.

<img src="./data/figures/Figure_1_pipeline.png" height="200"/>

## Example
```
Usage:
     sweeprecon -i path_to_nii <optional args>
     
     Optional arguments:
     --thickness <float>       :: thickness of acquisition slices in mm (default 4mm)
     --nstates <int>           :: number of respiration states to receonstruct to (defualt 4)
     --recon_thickness <float> :: thickness of reconstructed slices in mm (default isotropic)
     --redo                    :: flag to redo processing from scratch
     --resort_only             :: flag to stop processing after resort step
     --crop <float>            :: factor to crop from either side of image for bodyarea mask (default 0.2)
```

