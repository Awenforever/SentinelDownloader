import os
import zipfile
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, shape, box
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
import tempfile
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def read_geojson_from_file(geojson_path):
    """
    从本地文件读取 GeoJSON 并返回几何对象
    """
    try:
        gdf = gpd.read_file(geojson_path)
        print(f"成功读取 GeoJSON 文件，包含 {len(gdf)} 个要素")
        return gdf
    except Exception as e:
        print(f"读取 GeoJSON 文件失败: {e}")
        return None


def get_sentinel2_footprint_from_metadata(zip_path):
    """
    从 Sentinel-2 压缩包中的元数据文件获取精确的地理边界框
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 查找主元数据文件
            metadata_files = [f for f in zip_ref.namelist() if 'MTD_MSIL1C.xml' in f]

            if not metadata_files:
                print(f"在 {os.path.basename(zip_path)} 中未找到元数据文件")
                return None

            # 提取元数据文件到临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                metadata_path = zip_ref.extract(metadata_files[0], temp_dir)

                # 解析 XML 文件获取地理边界
                tree = ET.parse(metadata_path)
                root = tree.getroot()

                # 命名空间处理
                ns = {'n1': 'https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd'}

                # 查找地理边界坐标
                try:
                    # 方法1: 从Product_Footprint获取
                    footprint_elem = root.find('.//n1:Product_Footprint', ns)
                    if footprint_elem is not None:
                        coords_elem = footprint_elem.find('.//n1:EXT_POS_LIST', ns)
                        if coords_elem is not None:
                            coords_text = coords_elem.text.strip()
                            coords = [float(x) for x in coords_text.split()]

                            # 创建多边形
                            polygon_coords = []
                            for i in range(0, len(coords), 2):
                                if i + 1 < len(coords):
                                    polygon_coords.append((coords[i], coords[i + 1]))

                            if len(polygon_coords) >= 3:
                                footprint = Polygon(polygon_coords)
                                print(f"从元数据提取精确边界框: {footprint.bounds}")
                                return footprint

                    # 方法2: 从Geographic_Information获取（备用）
                    geo_info = root.find('.//n1:Geographic_Information', ns)
                    if geo_info is not None:
                        # 尝试不同的坐标元素路径
                        for elem_name in ['n1:Size', 'n1:Geoposition', 'n1:Product_Footprint']:
                            elem = geo_info.find(elem_name, ns)
                            if elem is not None:
                                # 这里简化处理，实际应该解析具体坐标
                                pass

                    # 方法3: 如果上述方法失败，使用简化方法从文件名提取瓦片信息
                    filename = os.path.basename(zip_path)
                    parts = filename.split('_')
                    if len(parts) >= 6:
                        tile_id = parts[5]
                        print(f"使用瓦片编号作为备用方法: {tile_id}")
                        # 这里可以添加瓦片编号到边界框的转换
                        return None

                except Exception as e:
                    print(f"解析地理边界时出错: {e}")
                    return None

    except Exception as e:
        print(f"读取 {zip_path} 元数据时出错: {e}")
        return None


def is_in_roi(footprint, roi_gdf):
    """
    检查 Sentinel-2 影像边界是否与任何感兴趣区域相交
    """
    if footprint is None:
        return False

    # 创建包含footprint的GeoDataFrame
    footprint_gdf = gpd.GeoDataFrame([1], geometry=[footprint], crs=roi_gdf.crs)

    # 检查空间相交
    intersection = gpd.overlay(footprint_gdf, roi_gdf, how='intersection')

    return len(intersection) > 0


def extract_swir_bands(zip_path, output_dir):
    """
    从 Sentinel-2 压缩包中提取 SWIR 波段 (B12, B8A, B4) 并生成预览图
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 查找波段文件
            band_files = {}
            band_patterns = {
                'B12': 'B12.jp2',
                'B8A': 'B8A.jp2',
                'B4': 'B04.jp2'
            }

            for band_name, pattern in band_patterns.items():
                band_file = [f for f in zip_ref.namelist() if pattern in f and 'IMG_DATA' in f]
                if band_file:
                    band_files[band_name] = band_file[0]

            if len(band_files) != 3:
                print(f"在 {os.path.basename(zip_path)} 中未找到所有SWIR波段")
                return None

            # 创建临时目录用于提取文件
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_files = {}

                # 提取波段文件
                for band_name, band_path in band_files.items():
                    extracted_path = zip_ref.extract(band_path, temp_dir)
                    extracted_files[band_name] = extracted_path

                # 读取波段数据
                bands_data = {}
                for band_name, file_path in extracted_files.items():
                    with rasterio.open(file_path) as src:
                        bands_data[band_name] = src.read(1)

                # 生成SWIR合成影像
                filename = os.path.basename(zip_path).replace('.zip', '')
                output_path = os.path.join(output_dir, f"{filename}_SWIR_preview.png")

                # 归一化并合成波段 (B12, B8A, B4) -> (R, G, B)
                def normalize_band(band):
                    band = band.astype(np.float32)
                    p2, p98 = np.percentile(band, (2, 98))
                    band = np.clip(band, p2, p98)
                    band = (band - p2) / (p98 - p2)
                    return band

                # 创建RGB合成
                red = normalize_band(bands_data['B12'])  # SWIR-2 -> 红色
                green = normalize_band(bands_data['B8A'])  # 窄近红外 -> 绿色
                blue = normalize_band(bands_data['B4'])  # 红波段 -> 蓝色

                rgb = np.dstack((red, green, blue))
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

                # 保存预览图
                img = Image.fromarray(rgb)
                img.save(output_path)

                print(f"生成SWIR预览图: {output_path}")
                return output_path

    except Exception as e:
        print(f"提取SWIR波段时出错 {zip_path}: {e}")
        return None


def process_sentinel2_data(zip_directory, geojson_path, output_dir, delete_files=True):
    """
    主处理函数
    """
    # 从本地文件读取感兴趣区域
    print("读取 GeoJSON 文件...")
    roi_gdf = read_geojson_from_file(geojson_path)

    if roi_gdf is None:
        print("无法读取 GeoJSON 文件，程序退出")
        return [], []

    print(f"找到 {len(roi_gdf)} 个感兴趣区域")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有 Sentinel-2 压缩包
    zip_files = []
    for f in os.listdir(zip_directory):
        if f.endswith('.zip') and ('S2A_' in f or 'S2B_' in f or 'MSIL1C' in f):
            zip_files.append(f)

    print(f"找到 {len(zip_files)} 个 Sentinel-2 压缩包")

    files_to_keep = []
    files_to_delete = []

    # 处理每个压缩包
    for zip_file in zip_files:
        zip_path = os.path.join(zip_directory, zip_file)

        print(f"\n处理: {zip_file}")

        # 获取精确的地理边界
        footprint = get_sentinel2_footprint_from_metadata(zip_path)

        if footprint is None:
            print(f"无法获取 {zip_file} 的地理边界，将保留该文件")
            files_to_keep.append(zip_file)
            continue

        # 检查是否在感兴趣区域内
        if is_in_roi(footprint, roi_gdf):
            files_to_keep.append(zip_file)
            print(f"✓ {zip_file} 在感兴趣区域内")
        else:
            files_to_delete.append(zip_file)
            print(f"✗ {zip_file} 不在感兴趣区域内")

            # 为待删除文件生成SWIR影像
            print(f"为待删除文件生成SWIR影像...")
            swir_path = extract_swir_bands(zip_path, output_dir)
            if swir_path:
                print(f"SWIR影像已保存: {swir_path}")

    # 删除不在感兴趣区域内的文件
    if delete_files and files_to_delete:
        print(f"\n删除 {len(files_to_delete)} 个文件...")
        for file_to_delete in files_to_delete:
            file_path = os.path.join(zip_directory, file_to_delete)
            try:
                os.remove(file_path)
                print(f"已删除: {file_to_delete}")
            except Exception as e:
                print(f"删除 {file_to_delete} 时出错: {e}")
    else:
        print(f"\n模拟模式 - 将删除 {len(files_to_delete)} 个文件:")
        for file_to_delete in files_to_delete:
            print(f"  - {file_to_delete}")

    # 生成处理报告
    generate_report(files_to_keep, files_to_delete, output_dir)

    print(f"\n处理完成!")
    print(f"保留文件: {len(files_to_keep)}")
    print(f"删除文件: {len(files_to_delete)}")

    return files_to_keep, files_to_delete


def generate_report(keep_files, delete_files, output_dir):
    """
    生成处理报告
    """
    report_path = os.path.join(output_dir, "processing_report.csv")

    report_data = []
    for filename in keep_files:
        report_data.append({
            'filename': filename,
            'status': '保留',
            'reason': '在感兴趣区域内'
        })

    for filename in delete_files:
        report_data.append({
            'filename': filename,
            'status': '删除',
            'reason': '不在感兴趣区域内'
        })

    df = pd.DataFrame(report_data)
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"处理报告已保存: {report_path}")


def main():
    """
    主函数
    """
    # 配置路径
    sentinel2_directory = input("请输入 Sentinel-2 数据目录路径: ").strip()
    geojson_path = input("请输入 GeoJSON 文件路径: ").strip()
    output_dir = input("请输入输出目录路径 (用于保存SWIR影像和报告): ").strip()

    # 验证路径是否存在
    if not os.path.exists(sentinel2_directory):
        print(f"错误: Sentinel-2 数据目录不存在: {sentinel2_directory}")
        return

    if not os.path.exists(geojson_path):
        print(f"错误: GeoJSON 文件不存在: {geojson_path}")
        return

    print(f"Sentinel-2 数据目录: {sentinel2_directory}")
    print(f"GeoJSON 文件: {geojson_path}")
    print(f"输出目录: {output_dir}")

    # 运行处理
    keep, delete = process_sentinel2_data(
        sentinel2_directory,
        geojson_path,
        output_dir,
        delete_files=False  # 设置为 True 来实际删除文件
    )


if __name__ == "__main__":
    main()