import argparse
import os
import numpy as np
import scipy.stats as spst
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix
import pandas as pd
import porespy as ps
from skimage.measure import block_reduce

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
VN = N_Voxel / 2        
VL = L_Voxel * 2   
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

###################################################
import psutil
import threading
import time

# Record maximum memory and CPU usage as global variables
def monitor_resources(interval=0.1):
    global max_mem, max_cpu, running
    process = psutil.Process()
    while running:
        # Memory (bytes)
        mem = process.memory_info().rss
        if mem > max_mem:
            max_mem = mem

        # CPU (%)
        cpu = process.cpu_percent(interval=None)  # instant CPU usage
        if cpu > max_cpu:
            max_cpu = cpu

        time.sleep(interval)       
####################################################
max_mem = 0
max_cpu = 0
running = True
# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

# ---- Generate polydisperse spheres ----
start_time = time.time()
print("Generating polydisperse spheres...")
im1 = ps.generators.polydisperse_spheres(shape=shape_ori, porosity=porosity, dist=dist, r_min=r_min, seed=seed_value)
elapsed_time = time.time() - start_time
print(f"Time required for generating polydisperse spheres: {elapsed_time} seconds\n")

# Stop monitoring
running = False
monitor_thread.join()

print(f"Peak memory usage: {max_mem / (1024**2):.2f} MB "
      f"({max_mem / (1024**3):.2f} GB)")
print(f"Peak CPU usage: {max_cpu:.1f}%\n")
####################################################
# ---- Binning to smooth the structure ----
max_mem = 0
max_cpu = 0
running = True
# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

def binarize_and_binning(image):
    VN_binning = int(N_Voxel / VN)
    binned_data_mean = block_reduce(image, block_size=(VN_binning, VN_binning, VN_binning), func=np.mean)
    return (binned_data_mean > 0.5).astype(int)  

start_time = time.time()
print("Starting binning to smooth the structure") 
im1 = np.abs(im1 - 1)  
LI_data = (binarize_and_binning(im1) * 255).astype(np.uint8)     
del im1

elapsed_time = time.time() - start_time
print(f"Time required for binning to smooth the structure: {elapsed_time} seconds\n")
# Stop monitoring
running = False
monitor_thread.join()

print(f"Peak memory usage: {max_mem / (1024**2):.2f} MB "
      f"({max_mem / (1024**3):.2f} GB)")
print(f"Peak CPU usage: {max_cpu:.1f}%\n")
####################################################
LI_data = (1 - (LI_data // 255)).astype(np.uint8)
start_time = time.time()
max_mem = 0
max_cpu = 0
running = True
# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

# ---- Compute tortuosity ----
print("Computing tortuosity...")

class CGSolver:
    def solve(self, A, b):
        A = csr_matrix(A)  
        b = b.ravel()  
        solution, info = spla.cg(A, b)  
        return solution, info  

solver = CGSolver()

results = ps.simulations.tortuosity_fd(LI_data, axis=2, solver=solver) 
elapsed_time = time.time() - start_time
print(f"Time required for tortuosity calculation: {elapsed_time} seconds\n")

# Stop monitoring
running = False
monitor_thread.join()

print(f"Peak memory usage: {max_mem / (1024**2):.2f} MB "
      f"({max_mem / (1024**3):.2f} GB)")
print(f"Peak CPU usage: {max_cpu:.1f}%\n")
#################################################################
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

##########################################################
