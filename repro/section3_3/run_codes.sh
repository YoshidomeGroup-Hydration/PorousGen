#!/bin/bash

# --- Parameter Settings ---
eta_Box=34        # \eta_Box [%]
L_Voxel=5         # L_Voxel [nm]
N_Voxel=1000        # N_Voxel
mean=45          # D_Mean [nm]
stan=1            # D_SD [nm]
N=6               # The box is divided into N^3 subdomains for parallel computation.
delta_VN=50       # l = L_Voxel * delta_VN [nm]
dL2=$(( mean + 2*stan )) # 2dL [nm]
phi_particle=50   # \phi_Particle [%]
eta_err=1         # \eta_Err [%]
ran=1             # seed

# L_Box = L_Voxel * N_Voxel 
# l = L_Voxel * delta_VN

# ---- Execute Python code ----
python3 code_PorousGen.py  \
    --eta_Box "$eta_Box" \
    --L_Voxel "$L_Voxel" \
    --N_Voxel "$N_Voxel" \
    --mean "$mean" \
    --stan "$stan" \
    --N "$N" \
    --delta_VN "$delta_VN" \
    --dL2 "$dL2"\
    --phi_particle "$phi_particle" \
    --eta_err "$eta_err" \
    --ran "$ran" \
    > log.txt
mkdir -p results/PorousGen/
mv temp/* results/PorousGen/
mv log.txt results/PorousGen/
rm -r temp
#################################################################
delta_VN=0
python3 code_PoreSpy_binning.py \
    --eta_Box "$eta_Box" \
    --L_Voxel "$L_Voxel" \
    --N_Voxel "$N_Voxel" \
    --mean "$mean" \
    --stan "$stan" \
    --ran "$ran" \
    > log.txt
mkdir -p results/PoreSpy_binning/
mv temp/* results/PoreSpy_binning/
mv log.txt results/PoreSpy_binning/
rm -r temp
