import numpy as np
import os
from multiprocessing import Pool, cpu_count
from scipy.spatial import KDTree
import tifffile as tiff
import scipy.stats as spst
from skimage.measure import block_reduce

# --- Parameter Settings ---
VN_ori = 200    # N_Box
VL_ori = 10     # L_Box [nm]
ran = 1         # seed
mean = 140      # D_Mean [nm]
stan = 1        # D_SD [nm]
eta_target = 40 # \eta_Box [%]
delta_VN = 50   # l = VL_ori * delta_VN [\mu m]
N = 5           # The box is divided into N^3 subdomains for parallel computation.

# --- Other Settings ---
VN = VN_ori / 2        
VL = VL_ori * 2         
phi_target = 100 - eta_target 
VN = int(VN_ori / 2)
VN_ori_add = VN_ori + delta_VN
subL = VN_ori_add // N
VN_ori_binned = VN_ori // 2
target_phi = phi_target/100
sub_phi = target_phi 
err_para = 0.01
domain_volume = VN**3
max_overlap_ratio = 0.5

output_directory = f"Structure_ran{ran}_mean{mean}_stan{stan}_phi{phi_target}"
os.makedirs(output_directory, exist_ok=True)
print(output_directory)

# --- Log-Normal Particle Size Distribution Setup ---
pixel_size_nm = VL_ori
mean_radius_px = 0.5 * (mean / pixel_size_nm)
stddev_radius_px = 0.5 * (stan / pixel_size_nm)
max_radius = int(mean_radius_px + 2*stddev_radius_px) 
buffer_size = int(2 * max_radius)
lower_limit_px = 5 / pixel_size_nm
upper_limit_px = 200 / pixel_size_nm

# --- Log-Normal Distribution Parameter Calculation ---
s = np.sqrt(np.log(1 + (stddev_radius_px / mean_radius_px) ** 2))
scale = mean_radius_px / np.exp(s ** 2 / 2)
dist = spst.lognorm(s=s, scale=scale)

min_R = lower_limit_px / 2
region_size = VN_ori // N
max_tries = 100000

############################################


# --- Function Definitions ---

def volume_sphere(r):
    return 4/3 * np.pi * r**3

def sample_radius(rng):
    diameter = dist.rvs(random_state=rng)
    radius = max(min_R, diameter)
    return radius

def sphere_overlap_volume(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return volume_sphere(min(r1, r2))
    part1 = (r1 + r2 - d)**2
    part2 = d**2 + 2*d*r2 - 3*r2**2 + 2*d*r1 + 6*r1*r2 - 3*r1**2
    return np.pi * part1 * part2 / (12 * d)

def can_place_kdtree(x, y, z, r, positions, radii, kdtree, radius_max, max_overlap_ratio):
    search_radius = r + radius_max
    idx_list = kdtree.query_ball_point([x, y, z], r=search_radius)
    for idx in idx_list:
        px, py, pz = positions[idx]
        pr = radii[idx]
        d = np.linalg.norm([x - px, y - py, z - pz])
        r_eff = min(r, pr)
        overlap = sphere_overlap_volume(r_eff, r_eff, d)
        if overlap / volume_sphere(r) > max_overlap_ratio:
            return False
    return True

def add_sphere(mask, x, y, z, r):
    Lz, Ly, Lx = mask.shape
    x_min = max(int(x - r), 0)
    x_max = min(int(x + r) + 1, Lx)
    y_min = max(int(y - r), 0)
    y_max = min(int(y + r) + 1, Ly)
    z_min = max(int(z - r), 0)
    z_max = min(int(z + r) + 1, Lz)

    zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
    sphere = (xx - x)**2 + (yy - y)**2 + (zz - z)**2 <= r**2
    mask[z_min:z_max, y_min:y_max, x_min:x_max][sphere] = True

def calc_phi(mask):
    return np.sum(mask) / (mask.size)

def fill_subdomain_with_buffer(args):
    i, j, k, subL, target_phi, buffer_size, ran = args
    extL = subL + 2 * buffer_size
    ext_mask = np.zeros((extL, extL, extL), dtype=bool)
    ext_positions = []
    ext_radii = []
    rng = np.random.default_rng(seed=ran * (10000 * i + 100 * j + k))
    kdtree = None
    update_kdtree_every = 10
    count_since_update = 0
    while calc_phi(ext_mask) < target_phi:
        r = sample_radius(rng)
        x = rng.integers(int(r), extL - int(r))
        y = rng.integers(int(r), extL - int(r))
        z = rng.integers(int(r), extL - int(r))
        if len(ext_positions) == 0:
            ok = True
        else:
            if kdtree is None or count_since_update == 0:
                kdtree = KDTree(ext_positions)
            ok = can_place_kdtree(x, y, z, r, ext_positions, ext_radii, kdtree, max_radius, max_overlap_ratio)
        if ok:
            ext_positions.append((x, y, z))
            ext_radii.append(r)
            add_sphere(ext_mask, x, y, z, r)
            count_since_update = (count_since_update + 1) % update_kdtree_every
        else:
            count_since_update = (count_since_update + 1) % update_kdtree_every
            
    inner_positions = []
    inner_radii = []
    for (x, y, z), r in zip(ext_positions, ext_radii):
        if buffer_size <= x < buffer_size + subL and buffer_size <= y < buffer_size + subL and buffer_size <= z < buffer_size + subL:
            gx = x - buffer_size + i * subL
            gy = y - buffer_size + j * subL
            gz = z - buffer_size + k * subL
            inner_positions.append((gx, gy, gz))
            inner_radii.append(r)
    return ext_mask, inner_positions, inner_radii

def calc_mask_from_particles(positions, radii, L):
    mask = np.zeros((L, L, L), dtype=bool)
    for (x, y, z), r in zip(positions, radii):
        add_sphere(mask, x, y, z, r)
    return mask

def thinning_weighted(positions, radii, actual_phi, target_phi, alpha, rng):
    if actual_phi <= target_phi:
        return positions, radii
    keep_ratio = target_phi / actual_phi
    volumes = volume_sphere(radii)
    total_volume = np.sum(volumes)
    volume_ratios = volumes / total_volume
    keep_probs = keep_ratio * (1 - alpha * volume_ratios)
    keep_probs = np.clip(keep_probs, 0, 1)
    mask = rng.random(len(positions)) < keep_probs
    return positions[mask], radii[mask]

def add_particles_until_phi_batch(positions, radii, target_phi, L, max_radius, max_overlap_ratio, rng, batch_size=10):
    new_positions = positions.tolist()
    new_radii = radii.tolist()
    mask = calc_mask_from_particles(positions, radii, L)
    current_phi = np.sum(mask) / (L**3)

    attempts = 0
    max_attempts = 10000

    while current_phi < target_phi and attempts < max_attempts:
        if len(new_positions) > 0:
            kdtree = KDTree(new_positions)
        else:
            kdtree = None

        candidates = []
        for _ in range(batch_size):
            r = sample_radius(rng)
            x = rng.integers(int(r), L - int(r))
            y = rng.integers(int(r), L - int(r))
            z = rng.integers(int(r), L - int(r))
            candidates.append((x, y, z, r))

        for (x, y, z, r) in candidates:
            if kdtree is None:
                ok = True
            else:
                ok = can_place_kdtree(x, y, z, r, new_positions, new_radii, kdtree, max_radius, max_overlap_ratio)

            if ok:
                new_positions.append((x, y, z))
                new_radii.append(r)
                add_sphere(mask, x, y, z, r)
                current_phi = np.sum(mask) / (L**3)
                if current_phi >= target_phi:
                    break
        attempts += batch_size
    return np.array(new_positions), np.array(new_radii)

def get_central_cut_mask(mask, VN_ori_add, VN_ori):
    center = VN_ori_add // 2
    half = VN_ori // 2
    return mask[
        center - half:center + half,
        center - half:center + half,
        center - half:center + half
    ]

def binarize_and_binning(image):
    VN_binning = int(VN_ori / VN)
    binned_data_mean = block_reduce(image, block_size=(VN_binning, VN_binning, VN_binning), func=np.mean)
    return (binned_data_mean > 0.5).astype(int)    



# --- Main Process ---

print("\nStarting initial particle placement")

args_list = [(i, j, k, subL, sub_phi, buffer_size, ran) for i in range(N) for j in range(N) for k in range(N)]

process_count = min(len(args_list), cpu_count())
with Pool(processes=process_count) as pool:
    results_iter = pool.imap_unordered(fill_subdomain_with_buffer, args_list)
    results = []
    total = len(args_list)
    for idx, res in enumerate(results_iter, 1):
        print(f"Overall progress: {idx}/{total} subregions processed")
        results.append(res)

all_positions = []
all_radii = []

for ext_mask, positions, radii in results:
    all_positions.extend(positions)
    all_radii.extend(radii)

all_positions = np.array(all_positions)
all_radii = np.array(all_radii)

max_iter = 100
positions = all_positions
radii = all_radii

idx_sort = np.lexsort((positions[:,2], positions[:,1], positions[:,0]))
positions = positions[idx_sort]
radii = radii[idx_sort]

mask = calc_mask_from_particles(positions, radii, VN_ori_add)
cut_mask = get_central_cut_mask(mask, VN_ori_add, VN_ori)

print("Starting binning")
cut_mask_binned =binarize_and_binning(cut_mask)
phi_current = calc_phi(cut_mask_binned)


##########################################################

print("Starting volume fraction adjustment")

for iteration in range(max_iter):
    mask = calc_mask_from_particles(positions, radii, VN_ori_add)
    cut_mask = get_central_cut_mask(mask, VN_ori_add, VN_ori)
    cut_mask_binned =binarize_and_binning(cut_mask)
    actual_phi = np.sum(cut_mask_binned) / domain_volume
    err = abs(actual_phi - target_phi) / target_phi
    print(f"[Iter {iteration+1}] Volume fraction: {actual_phi:.4f}, Error: {err*100:.2f}%")

    if abs(phi_current - target_phi) < err_para:
        break
    if phi_current > target_phi:
        rng_thinning = np.random.default_rng( ran * 12345 + iteration )
        positions, radii = thinning_weighted(positions, radii, phi_current, target_phi, alpha=0, rng=rng_thinning) 
        mask = calc_mask_from_particles(positions, radii, VN_ori_add)
        cut_mask = get_central_cut_mask(mask, VN_ori_add, VN_ori)
        cut_mask_binned =binarize_and_binning(cut_mask)
        phi_current = calc_phi(cut_mask_binned)        
    else:
        rng_add = np.random.default_rng(ran + 1000 + iteration)
        positions, radii = add_particles_until_phi_batch(positions, radii, target_phi, VN, max_radius, max_overlap_ratio, rng=rng_add)
        mask = calc_mask_from_particles(positions, radii, VN_ori_add)
        cut_mask = get_central_cut_mask(mask, VN_ori_add, VN_ori)
        cut_mask_binned =binarize_and_binning(cut_mask)
        phi_current = calc_phi(cut_mask_binned)
        
mask = calc_mask_from_particles(positions, radii, VN_ori_add)
cut_mask = get_central_cut_mask(mask, VN_ori_add, VN_ori)
final_mask = binarize_and_binning(cut_mask)
actual_phi = calc_phi(final_mask)

print(f"\nFinal volume fraction after adjustment: {actual_phi:.4f}")

# Output size information
print("\n--- Size Information ---")
print(f"Overall size L × L × L                     : {VN_ori_add} × {VN_ori_add} × {VN_ori_add}")
print(f"Subregion size subL × subL × subL          : {subL} × {subL} × {subL}")
print(f"Extended size (per subregion)              : {subL + 2*buffer_size} × {subL + 2*buffer_size} × {subL + 2*buffer_size}")
print(f"Combined mask size (excluding buffer)     : {cut_mask.shape}")
print(f"Combined binned mask size (excluding buffer): {final_mask.shape}")
print(f"Buffer width                               : {buffer_size}")
print("---------------------\n")


def save_tiff_stack(image, output_dir):
    for z in range(image.shape[2]):
        slice_data = image[:, :, z]
        file_path = os.path.join(output_dir, f"normalZ{z+1:05d}.tif")
        tiff.imwrite(file_path, slice_data.astype(np.uint8))


save_tiff_stack(final_mask* 255, output_directory)
