##########################################################
# code_pic.py
##########################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
    
    # 中央点をNaNに設定（事前にcenter_x, center_yを定義する必要がある）
    center_x = central_slice.shape[0] // 2
    center_y = central_slice.shape[1] // 2
    central_slice[center_x, center_y] = np.nan

    # 画像の表示
    im = ax.imshow(central_slice, cmap='viridis', vmin=vmin, vmax=vmax, extent=[-VN/2, VN/2, -VN/2, VN/2])
    
    # カラーバーの作成とラベルのフォントサイズ設定
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)  # カラーバーのメモリのフォントサイズを設
    
    # カラーバーに「自然対数スケール」のラベルを追加
    cbar.set_label('Log scale', fontsize=16)  # または、cbar.set_label('自然対数スケール', fontsize=16)
    
    # カラーバーの作成
    del_k = 2 * np.pi / (VL*VN)  # 逆空間の間隔（単位：nm^-1）
       
    # 軸ラベルの設定
    ax.set_xlabel("$k_x \\;  [\\mathrm{nm}^{-1}]$", fontsize=16)
    ax.set_ylabel("$k_y \\;  [\\mathrm{nm}^{-1}]$", fontsize=16)

    # メモリを0を中心に設定
    num_ticks = 5  # 軸に表示するメモリの数
    xticks = np.linspace(-VN/2, VN/2, num_ticks)
    yticks = np.linspace(-VN/2, VN/2, num_ticks)

    # FixedLocator を使用して軸の位置を設定
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # VLをかけて整数に変換したラベルを設定
    ax.set_xticklabels([f"{tick * del_k:.2f}" for tick in xticks])
    ax.set_yticklabels([f"{tick * del_k:.2f}" for tick in yticks])

    # FixedLocator と FixedFormatter を一緒に使用
    ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))

    # メモリのフォントサイズを変更
    ax.tick_params(axis='both', which='major', labelsize=16)

    
    # 画像保存
    plt.savefig(os.path.join(output_directory, f"{base_name}.png"))
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
