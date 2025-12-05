##########################################################
# code_pic.py
##########################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

        ax.set_xlabel(r"$x \, [\mathrm{\mu m}]$", fontsize=16)
        ax.set_ylabel(r"$y \, [\mathrm{\mu m}]$", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
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
        plt.savefig(save_path)
        plt.close()
