# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date     : 2025/10/14 14:09 
@Author   : Jinhong Wu
@Contact  : vive@mail.ustc.edu.cn
@Project  : SpectralGPT
@File     : visualize_aot.py
@IDE      : PyCharm

Copyright (c) 2025 Jinhong Wu. All rights reserved.
"""
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os


def visualize_aot_sentinel2(l2a_product_path, output_path=None):
    """
    可视化Sentinel-2 L2A产品的AOT数据

    参数:
    l2a_product_path: L2A产品根目录路径
    output_path: 输出图像路径(可选)
    """

    # 查找AOT文件
    aot_files = []
    for root, dirs, files in os.walk(l2a_product_path):
        for file in files:
            if file.endswith('AOT_10m.jp2'):
                aot_files.append(os.path.join(root, file))

    if not aot_files:
        print("未找到AOT文件")
        return

    # 选择最高分辨率的AOT数据
    aot_files.sort(key=lambda x: int(x.split('_')[-1].split('m')[0]))
    aot_file = aot_files[0]  # 最高分辨率

    print(f"使用AOT文件: {aot_file}")

    # 读取AOT数据
    with rasterio.open(aot_file) as src:
        aot_data = src.read(1)
        profile = src.profile

        # 获取无数据值
        nodata = src.nodata
        if nodata is not None:
            aot_data = np.ma.masked_where(aot_data == nodata, aot_data)

        # 获取坐标信息
        bounds = src.bounds

    # 创建自定义色彩映射 (蓝-绿-黄-红)
    colors = ['blue', 'cyan', 'green', 'yellow', 'orange', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('aot_cmap', colors, N=256)

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1: 基础AOT可视化
    im1 = ax1.imshow(aot_data, cmap='hot', vmin=236, vmax=560)
    ax1.set_title('Sentinel-2 AOT Distribution\n(Resolution: {}m)'.format(
        int(aot_file.split('_')[-1].split('m')[0])))
    ax1.axis('off')

    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Aerosol Optical Depth (AOD)', rotation=270, labelpad=15)

    # 子图2: 统计直方图
    valid_data = aot_data.compressed() if isinstance(aot_data, np.ma.MaskedArray) else aot_data
    if len(valid_data) > 0:
        ax2.hist(valid_data.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('AOT Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('AOT Distribution Statistics')
        ax2.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"""
        Statistical Information:
        Min: {valid_data.min():.3f}
        Max: {valid_data.max():.3f}
        Mean: {valid_data.mean():.3f}
        Std: {valid_data.std():.3f}
        # Valid Pixels: {len(valid_data):,}
        """
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_path}")

    plt.show()

    return aot_data, profile


# 使用示例
if __name__ == "__main__":
    # 替换为你的L2A产品路径
    l2a_path = r"D:\Downloads\Compressed\S2A_MSIL2A_20250813T112131_N9999_R037_T29TPG_20251014T084123.SAFE"

    # 可视化AOT
    aot_data, profile = visualize_aot_sentinel2(l2a_path, "aot_visualization.png")