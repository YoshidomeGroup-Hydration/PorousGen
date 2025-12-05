#!/bin/bash

# --- Parameter Settings ---
eta_Box=50        # \eta_Box [%]
L_Voxel=20         # L_Voxel [nm]
N_Voxel=500        # N_Voxel
mean=140          # D_Mean [nm]
stan=1            # D_SD [nm]
N=5               # The box is divided into N^3 subdomains for parallel computation.
delta_VN=50       # l = L_Voxel * delta_VN [nm]
dL2=$(( mean + 2*stan )) # 2dL [nm]
phi_particle=50   # \phi_Particle [%]
eta_err=1         # \eta_Err [%]
ran=1             # seed

# L_Box = L_Voxel * N_Voxel 
# l = L_Voxel * delta_VN

# ---- Execute Python code ----
python3 code_PoreSpy.py \
    --eta_Box "$eta_Box" \
    --L_Voxel "$L_Voxel" \
    --N_Voxel "$N_Voxel" \
    --mean "$mean" \
    --stan "$stan" \
    --ran "$ran" 

mkdir -p results/PoreSpy/
mv temp/* results/PoreSpy/
rm -r temp
#################################################################
eta_Box=46
python3 code_PorousGen_nobinning.py \
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
    --ran "$ran" 

mkdir -p results/PorousGen_nobinning/
mv temp/* results/PorousGen_nobinning/
rm -r temp