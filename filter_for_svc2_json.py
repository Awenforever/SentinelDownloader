#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Date     : 2025/10/15 11:37
@Author   : Jinhong Wu
@Contact  : vive@mail.ustc.edu.cn
@Project  : SpectralGPT
@File     : filter_for_svc2_json.py
@IDE      : PyCharm

Copyright (c) 2025 Jinhong Wu. All rights reserved.
"""
import pandas as pd
import json
from pathlib import Path
import numpy as np
from collections import defaultdict


def create_merged_grid_polygons(input_file, output_file, grid_size=0.1, lat_min=-60, lat_max=80, confidence_filter="h"):
    """
    创建合并的网格多边形，将相邻的网格合并成连续区域

    参数:
    - input_file: 输入JSON文件路径
    - output_file: 输出GeoJSON文件路径
    - grid_size: 网格大小(度)
    - lat_min: 最小纬度
    - lat_max: 最大纬度
    - confidence_filter: 置信度筛选，只保留指定置信度的数据
    """

    # 读取JSON数据
    df = pd.read_json(input_file)

    # 筛选纬度在指定范围内的数据
    filtered_df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)]
    print(f"纬度筛选完成，原始数据{len(df)}条，筛选后{len(filtered_df)}条")

    # 筛选置信度为"h"的数据
    if confidence_filter:
        filtered_df = filtered_df[filtered_df['confidence'] == confidence_filter]
        print(f"置信度筛选完成，保留置信度为'{confidence_filter}'的数据，筛选后{len(filtered_df)}条")

    # 如果没有数据满足条件，返回空结果
    if len(filtered_df) == 0:
        print("警告：没有数据满足筛选条件")
        # 创建空的GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"空结果已保存至 {output_file}")
        return []

    # 计算网格索引
    filtered_df['grid_lat'] = (filtered_df['latitude'] / grid_size).round().astype(int)
    filtered_df['grid_lon'] = (filtered_df['longitude'] / grid_size).round().astype(int)

    # 获取唯一的网格
    unique_grids = filtered_df[['grid_lat', 'grid_lon']].drop_duplicates()

    print(f"网格生成完成，网格大小: {grid_size}度, 网格数量: {len(unique_grids)}")

    # 将网格转换为集合以便快速查找
    grid_set = set(zip(unique_grids['grid_lat'], unique_grids['grid_lon']))

    # 使用深度优先搜索找到连通区域
    visited = set()
    regions = []

    for grid in grid_set:
        if grid not in visited:
            # 找到一个新的连通区域
            region = []
            stack = [grid]

            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    region.append(current)

                    # 检查四个方向的邻居
                    lat, lon = current
                    neighbors = [
                        (lat - 1, lon),  # 上
                        (lat + 1, lon),  # 下
                        (lat, lon - 1),  # 左
                        (lat, lon + 1)  # 右
                    ]

                    for neighbor in neighbors:
                        if neighbor in grid_set and neighbor not in visited:
                            stack.append(neighbor)

            regions.append(region)

    print(f"找到 {len(regions)} 个连通区域")

    # 为每个连通区域创建多边形
    features = []

    for region in regions:
        # 计算区域的边界
        min_lat = min(grid[0] for grid in region)
        max_lat = max(grid[0] for grid in region)
        min_lon = min(grid[1] for grid in region)
        max_lon = max(grid[1] for grid in region)

        # 创建边界框多边形
        lat_min_bound = min_lat * grid_size
        lat_max_bound = (max_lat + 1) * grid_size
        lon_min_bound = min_lon * grid_size
        lon_max_bound = (max_lon + 1) * grid_size

        # 创建多边形坐标（顺时针顺序，闭合多边形）
        coordinates = [
            [
                [lon_min_bound, lat_min_bound],  # 左下
                [lon_min_bound, lat_max_bound],  # 左上
                [lon_max_bound, lat_max_bound],  # 右上
                [lon_max_bound, lat_min_bound],  # 右下
                [lon_min_bound, lat_min_bound]  # 回到起点闭合
            ]
        ]

        feature = {
            "type": "Feature",
            "properties": {
                "grid_count": len(region),
                "min_lat": lat_min_bound,
                "max_lat": lat_max_bound,
                "min_lon": lon_min_bound,
                "max_lon": lon_max_bound,
                "confidence": confidence_filter
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        }

        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"结果已保存至 {output_file}")

    return regions


def create_concave_hull_polygons(input_file, output_file, grid_size=0.1, lat_min=-60, lat_max=80,
                                 confidence_filter="h"):
    """
    创建凹包多边形，更精确地勾勒连通区域的形状

    参数:
    - input_file: 输入JSON文件路径
    - output_file: 输出GeoJSON文件路径
    - grid_size: 网格大小(度)
    - lat_min: 最小纬度
    - lat_max: 最大纬度
    - confidence_filter: 置信度筛选，只保留指定置信度的数据
    """

    # 读取JSON数据
    df = pd.read_json(input_file)

    # 筛选纬度在指定范围内的数据
    filtered_df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)]
    print(f"纬度筛选完成，原始数据{len(df)}条，筛选后{len(filtered_df)}条")

    # 筛选置信度为"h"的数据
    if confidence_filter:
        filtered_df = filtered_df[filtered_df['confidence'] == confidence_filter]
        print(f"置信度筛选完成，保留置信度为'{confidence_filter}'的数据，筛选后{len(filtered_df)}条")

    # 如果没有数据满足条件，返回空结果
    if len(filtered_df) == 0:
        print("警告：没有数据满足筛选条件")
        # 创建空的GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        # 保存结果
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"空结果已保存至 {output_file}")
        return []

    # 计算网格索引
    filtered_df['grid_lat'] = (filtered_df['latitude'] / grid_size).round().astype(int)
    filtered_df['grid_lon'] = (filtered_df['longitude'] / grid_size).round().astype(int)

    # 获取唯一的网格
    unique_grids = filtered_df[['grid_lat', 'grid_lon']].drop_duplicates()

    print(f"网格生成完成，网格大小: {grid_size}度, 网格数量: {len(unique_grids)}")

    # 将网格转换为集合以便快速查找
    grid_set = set(zip(unique_grids['grid_lat'], unique_grids['grid_lon']))

    # 使用深度优先搜索找到连通区域
    visited = set()
    regions = []

    for grid in grid_set:
        if grid not in visited:
            # 找到一个新的连通区域
            region = []
            stack = [grid]

            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    region.append(current)

                    # 检查四个方向的邻居
                    lat, lon = current
                    neighbors = [
                        (lat - 1, lon),  # 上
                        (lat + 1, lon),  # 下
                        (lat, lon - 1),  # 左
                        (lat, lon + 1)  # 右
                    ]

                    for neighbor in neighbors:
                        if neighbor in grid_set and neighbor not in visited:
                            stack.append(neighbor)

            regions.append(region)

    print(f"找到 {len(regions)} 个连通区域")

    # 为每个连通区域创建凹包多边形
    features = []

    for region in regions:
        # 收集所有边界点
        boundary_points = set()

        for grid in region:
            lat, lon = grid

            # 检查每个网格的四个边，如果邻居不存在，则该边是边界
            # 上边
            if (lat - 1, lon) not in grid_set:
                boundary_points.add((lon * grid_size, (lat + 1) * grid_size))  # 左上
                boundary_points.add(((lon + 1) * grid_size, (lat + 1) * grid_size))  # 右上

            # 下边
            if (lat + 1, lon) not in grid_set:
                boundary_points.add((lon * grid_size, lat * grid_size))  # 左下
                boundary_points.add(((lon + 1) * grid_size, lat * grid_size))  # 右下

            # 左边
            if (lat, lon - 1) not in grid_set:
                boundary_points.add((lon * grid_size, lat * grid_size))  # 左下
                boundary_points.add((lon * grid_size, (lat + 1) * grid_size))  # 左上

            # 右边
            if (lat, lon + 1) not in grid_set:
                boundary_points.add(((lon + 1) * grid_size, lat * grid_size))  # 右下
                boundary_points.add(((lon + 1) * grid_size, (lat + 1) * grid_size))  # 右上

        # 将边界点转换为列表
        points = list(boundary_points)

        if len(points) < 3:
            # 如果点太少，使用边界框
            min_lat = min(grid[0] for grid in region)
            max_lat = max(grid[0] for grid in region)
            min_lon = min(grid[1] for grid in region)
            max_lon = max(grid[1] for grid in region)

            coordinates = [
                [
                    [min_lon * grid_size, min_lat * grid_size],
                    [min_lon * grid_size, (max_lat + 1) * grid_size],
                    [(max_lon + 1) * grid_size, (max_lat + 1) * grid_size],
                    [(max_lon + 1) * grid_size, min_lat * grid_size],
                    [min_lon * grid_size, min_lat * grid_size]
                ]
            ]
        else:
            # 使用简单的凸包算法（这里简化处理，实际可以使用更复杂的凹包算法）
            # 按角度排序点，形成多边形
            center_x = sum(p[0] for p in points) / len(points)
            center_y = sum(p[1] for p in points) / len(points)

            # 按相对于中心点的角度排序
            points_sorted = sorted(points, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))

            # 闭合多边形
            coordinates = [points_sorted + [points_sorted[0]]]

        feature = {
            "type": "Feature",
            "properties": {
                "grid_count": len(region),
                "confidence": confidence_filter
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        }

        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"结果已保存至 {output_file}")

    return regions


# 使用示例
if __name__ == "__main__":
    PATH = Path(r"D:\Downloads\Compressed\DL_FIRE_SV-C2_673856\fire_archive_SV-C2_673856.json")

    # 创建合并的网格多边形（只保留高置信度数据）
    regions = create_merged_grid_polygons(
        input_file=PATH,
        output_file=PATH.parent / 'merged_grid_polygons_high_confidence.geojson',
        grid_size=1,  # 1度网格，约111km x 111km
        lat_min=-60,
        lat_max=80,
        confidence_filter="h"  # 只保留高置信度数据
    )

    # 或者创建凹包多边形（更精确的形状）
    regions_concave = create_concave_hull_polygons(
        input_file=PATH,
        output_file=PATH.parent / 'concave_hull_polygons_high_confidence.geojson',
        grid_size=1,
        lat_min=-60,
        lat_max=80,
        confidence_filter="h"  # 只保留高置信度数据
    )