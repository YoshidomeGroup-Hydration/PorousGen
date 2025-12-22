##########################################################
# code_pic.py
##########################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.titlesize": 24,
    "legend.fontsize": 24,
})

# =======================================================
# Save the central slice as a 2D image.
# =======================================================
def plot_and_save_image(
        central_slice, VN, VL,
        vmin, vmax,
        output_directory, base_name
    ):
    
    #central_slice = central_slice.T
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    center_x = central_slice.shape[0] // 2
    center_y = central_slice.shape[1] // 2
    central_slice[center_x, center_y] = np.nan

    im = ax.imshow(central_slice, cmap='viridis', vmin=vmin, vmax=vmax, extent=[-VN/2, VN/2, -VN/2, VN/2])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log scale')  

    del_k = 2 * np.pi / (VL*VN)  # nm^-1
       
    ax.set_xlabel("$k_x \\;  [\\mathrm{nm}^{-1}]$")
    ax.set_ylabel("$k_y \\;  [\\mathrm{nm}^{-1}]$")

    num_ticks = 5  
    xticks = np.linspace(-VN/2, VN/2, num_ticks)
    yticks = np.linspace(-VN/2, VN/2, num_ticks)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([f"{tick * del_k:.2f}" for tick in xticks])
    ax.set_yticklabels([f"{tick * del_k:.2f}" for tick in yticks])

    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))

    ax.tick_params(axis='both', which='major')

    save_path = os.path.join(output_directory, f"{base_name}.png")
    plt.savefig(save_path, bbox_inches="tight")    
    plt.close()


# =======================================================
# Save all Z-direction slices as 2D images.
# =======================================================
def save_all_slices(
        data, VN, VL,
        output_directory, base_name
    ):

    data_shape = data.shape
    data_size = tuple(dim // 5 * 5 for dim in data_shape)
    nz = data_size[2]

    for k in range(nz):

        slice_2d = data[:data_size[0], :data_size[1], k]
        slice_2d = slice_2d.T
        slice_2d = np.where(slice_2d == 0, np.nan, slice_2d)

        fig, ax = plt.subplots(figsize=(10, 7))

        vmin, vmax = np.nanmin(slice_2d), np.nanmax(slice_2d)

        im = ax.imshow(slice_2d, cmap='bwr_r', vmin=vmin, vmax=vmax)

        ax.set_xlabel(r"$x \, [\mathrm{\mu m}]$")
        ax.set_ylabel(r"$y \, [\mathrm{\mu m}]$")
        ax.tick_params(axis='both', which='major')
        
        num_ticks = 6
        xticks = np.linspace(0, VN//5 * 5, num_ticks)
        yticks = np.linspace(0, VN//5 * 5, num_ticks)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{tick * VL / 1000:.1f}" for tick in xticks])
        ax.set_yticklabels([f"{tick * VL / 1000:.1f}" for tick in yticks])
        ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
        
        save_path = os.path.join(output_directory, f"{base_name}_slice{k:04d}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
