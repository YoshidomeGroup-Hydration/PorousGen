# PorousGen
“PorousGen” allows for the efficient generation of large-scale porous structures through parallel computation. It can also accurately create structures according to a defined target porosity.

# Requirement
Python 3.12.3~, numpy scipy tifffile scikit-image  

# Licence
“PorousGen” is available under the MIT License.

# Citing this work
If you use "PorousGen", please cite:
```
PorousGen: An efficient algorithm for generating porous structures with accurate porosity and uniform density distribution
Shota Arai and Takashi Yoshidome
arXiv: 2510.17133 (2025). http://arxiv.org/abs/2510.17133
```
# Contact
If you have any questions, please contact Shota Arai at<br>
shota.arai.c2@tohoku.ac.jp

# Usage

## Test Parameters

The following parameters are used for test execution:

```text
# --- Parameter Settings ---
VN_ori = 200        # N_Box
VL_ori = 10         # L_Box [nm]
ran = 1             # seed
mean = 140          # D_Mean [nm]
stan = 1            # D_SD [nm]
eta_target = 40     # \eta_Box [%]
delta_VN = 50       # l = VL_ori * delta_VN [\mu m]
N = 5               # The box is divided into N^3 subdomains for parallel computation.
eta_err = 1         # \eta_Err [%]
phi_particle = 50   # \phi_Particle [%]
dL2 = mean + 2*stan # 2dL [nm]
```

A 3D porous structure with 100^3 voxels of size 20 nm is generated.
TIFF images of the structure will be output.
