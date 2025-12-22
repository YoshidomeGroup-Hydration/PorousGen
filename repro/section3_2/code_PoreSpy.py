import argparse
import os
import numpy as np
import scipy.stats as spst
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
import pandas as pd
import porespy as ps

# ------------------------
# Argument parser
# ------------------------
parser = argparse.ArgumentParser(description="Porous structure generator parameters")
parser.add_argument("--eta_Box",       type=int, required=True, help="η_Box [%]")
parser.add_argument("--L_Voxel",       type=int, required=True, help="L_Voxel [nm]")
parser.add_argument("--N_Voxel",       type=int, required=True, help="N_Voxel")
parser.add_argument("--mean",          type=int, required=True, help="D_Mean [nm]")
parser.add_argument("--stan",          type=int, required=True, help="D_SD [nm]")
parser.add_argument("--ran",           type=int, required=True, help="seed")

args = parser.parse_args()

# ------------------------
# Parameter Settings
# ------------------------
eta_Box      = args.eta_Box
L_Voxel      = args.L_Voxel
N_Voxel      = args.N_Voxel
mean         = args.mean
stan         = args.stan
ran          = args.ran

print("### Parameters loaded ###")
print(vars(args))

output_directory = "temp"
os.makedirs(output_directory, exist_ok=True)
print(output_directory)

# --- Other Settings ---
VN = N_Voxel         
VL = L_Voxel    
pixel_size_nm = L_Voxel  
shape_ori = [N_Voxel, N_Voxel, N_Voxel]  
porosity = eta_Box/100  
mean_diameter_nm_dummy = mean  
mean_diameter_nm = mean_diameter_nm_dummy * 0.5
stddev_diameter_nm = stan  
r_min = 1
lower_limit_nm = 35  
upper_limit_nm = 200  
seed_value = ran 

# ---- Convert to pixel units ----
mean_diameter_px = mean_diameter_nm / pixel_size_nm
stddev_diameter_px = stddev_diameter_nm / pixel_size_nm
lower_limit_px = lower_limit_nm / pixel_size_nm
upper_limit_px = upper_limit_nm / pixel_size_nm

# ---- Generate distributions ----
s = np.sqrt(np.log(1 + (stddev_diameter_px / mean_diameter_px) ** 2))
scale = mean_diameter_px / np.exp(s**2 / 2)
dist = spst.lognorm(s=s, scale=scale)

# ---- Generate polydisperse spheres ----
print("Generating polydisperse spheres...")
im1 = ps.generators.polydisperse_spheres(shape=shape_ori, porosity=porosity, dist=dist, r_min=r_min, seed=seed_value)
"""
# ---- Compute tortuosity ----
print("Computing tortuosity...")

class CGSolver:
    def solve(self, A, b):
        A = csr_matrix(A)  
        b = b.ravel()  
        solution, info = spla.cg(A, b)  
        return solution, info  

solver = CGSolver()

results = ps.simulations.tortuosity_fd(im1, axis=2, solver=solver) 

formation_factor = results.formation_factor
original_porosity = results.original_porosity
effective_porosity = results.effective_porosity
print("Formation Factor:", formation_factor)
print("original_porosity:", original_porosity)
print("effective_porosity:", effective_porosity)

# Compile results into a DataFrame
df = pd.DataFrame({
    "Formation Factor": [formation_factor],
    "Original Porosity": [original_porosity],
    "Effective Porosity": [effective_porosity]
})

# Save in proper .xlsx format
filename = os.path.join(output_directory, f"results.xlsx")
df.to_excel(filename, index=False, engine="openpyxl")
"""
##########################################################
from helpers import plot_and_save_image, save_all_slices
from scipy.fft import fftn, ifftn, fftshift

data = np.abs(im1 - 1) * 255
central_slice = data[data.shape[0]//2, :, :]


# ---- Save all Z-direction slices as 2D images ----
output_dir = os.path.join(output_directory, "electron_density")
os.makedirs(output_dir, exist_ok=True)
save_all_slices(data, VN=VN, VL=VL, output_directory=output_dir, base_name="3D")

g = fftn(data)
PS_3d = np.abs(g) ** 2
data = np.fft.fftshift(PS_3d)
#data = data.T
central_slice = data[:, data.shape[0]//2,  :]
central_slice = np.log(central_slice + 1e-10)

# ---- Save the central slice as a 2D image  ----
output_dir = os.path.join(output_directory, "power_spectrum")
os.makedirs(output_dir, exist_ok=True)
plot_and_save_image(
    central_slice,
    VN=VN,
    VL=VL,
    vmin=15,
    vmax=32,
    output_directory=output_dir,
    base_name="PS",
)
