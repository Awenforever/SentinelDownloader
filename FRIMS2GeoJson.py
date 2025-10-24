import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')


class FireDataProcessor:
    def __init__(self, input_dir, output_dir, k=10, grid_size=0.1):
        """
        初始化火灾数据处理类

        参数:
        input_dir: 国家CSV文件输入目录
        output_dir: 处理结果输出目录
        k: 每月选取的top-k密度区域数量
        grid_size: 空间网格大小(度)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.k = k
        self.grid_size = grid_size
        os.makedirs(output_dir, exist_ok=True)

    def process_country_data(self):
        """处理单个国家CSV文件，进行筛选和精简"""
        country_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        if not country_files:
            print(f"在 {self.input_dir} 中没有找到CSV文件")
            return None

        all_dfs = []
        total_before = 0
        total_after = 0

        for file in tqdm(country_files, desc="处理国家文件"):
            try:
                # 读取CSV文件（只加载必要列以节省内存）
                df = pd.read_csv(file, usecols=['latitude', 'longitude', 'acq_date', 'confidence', 'daynight'])
                country_name = os.path.basename(file).split('.')[0]

                # 记录筛选前的数量
                count_before = len(df)
                total_before += count_before

                # 筛选条件
                df = df[(df['confidence'] == 'h') &
                        (df['daynight'] == 'D') &
                        (df['latitude'].between(-60, 75))]  # 排除两极和海洋

                # 只保留必要列
                df = df[['latitude', 'longitude', 'acq_date']]

                # 记录筛选后的数量
                count_after = len(df)
                total_after += count_after

                tqdm.write(f"{country_name}: 筛选前 {count_before} 个点, 筛选后 {count_after} 个点")
                all_dfs.append(df)

            except Exception as e:
                print(f"处理 {file} 时出错: {e}")

        # 合并所有国家数据
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n全局统计: 筛选前总数 {total_before} 个点, 筛选后总数 {total_after} 个点")
        return combined_df

    def process_monthly_data(self, df):
        """按月处理数据并生成GeoJSON"""
        # 转换日期并提取月份
        df['acq_date'] = pd.to_datetime(df['acq_date'])
        df['month'] = df['acq_date'].dt.month_name().str[:3].str.lower()

        # 按月分组处理
        for month, group in tqdm(df.groupby('month'), desc="处理月度数据"):
            points = group[['longitude', 'latitude']].values

            if len(points) == 0:
                tqdm.write(f"{month.capitalize()}月没有数据，跳过")
                continue

            # 创建空间网格
            grid = self.create_spatial_grid(points)

            if len(grid) < self.k:
                tqdm.write(f"{month.capitalize()}月只有 {len(grid)} 个网格，小于 k={self.k}")
                k = len(grid)
            else:
                k = self.k

            # 获取top-k密集区域
            top_grids = grid.nlargest(k, 'count')

            # 生成多边形
            polygons = []
            for _, row in top_grids.iterrows():
                minx, miny = row['minx'], row['miny']
                maxx, maxy = minx + self.grid_size, miny + self.grid_size
                polygons.append(Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]))

            # 创建multipolygon
            multipoly = self.create_multipolygon(polygons)

            # 保存GeoJSON
            self.save_geojson(multipoly, month)

    def create_spatial_grid(self, points):
        """创建空间网格并统计点密度"""
        # 创建网格边界
        minx, miny = np.floor(points.min(axis=0) / self.grid_size) * self.grid_size
        maxx, maxy = np.ceil(points.max(axis=0) / self.grid_size) * self.grid_size

        # 生成网格
        x_bins = np.arange(minx, maxx + self.grid_size, self.grid_size)
        y_bins = np.arange(miny, maxy + self.grid_size, self.grid_size)

        # 统计每个网格中的点数
        grid_counts, x_edges, y_edges = np.histogram2d(
            points[:, 0], points[:, 1], bins=[x_bins, y_bins]
        )

        # 转换为DataFrame
        grid_data = []
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                count = grid_counts[i, j]
                if count > 0:
                    grid_data.append({
                        'minx': x_edges[i],
                        'miny': y_edges[j],
                        'count': count
                    })

        return pd.DataFrame(grid_data)

    def create_multipolygon(self, polygons):
        """创建符合要求的多边形，确保面积不超过阈值"""
        MAX_AREA = 5e5  # 最大面积阈值 (km²)，约等于西班牙/法国大小

        # 计算每个多边形的面积（使用近似计算）
        def calc_approx_area(poly):
            coords = np.array(poly.exterior.coords)
            lons, lats = coords[:, 0], coords[:, 1]
            avg_lat = np.mean(lats)
            lon_scale = np.cos(np.radians(avg_lat))
            area = (np.max(lons) - np.min(lons)) * lon_scale * 111 * \
                   (np.max(lats) - np.min(lats)) * 111
            return area

        # 合并相邻多边形
        merged = unary_union(polygons)

        # 如果合并后是单个多边形且面积过大，则分割
        if isinstance(merged, Polygon):
            area = calc_approx_area(merged)
            if area > MAX_AREA:
                # 简单分割为4个小多边形
                minx, miny, maxx, maxy = merged.bounds
                midx, midy = (minx + maxx) / 2, (miny + maxy) / 2
                polygons = [
                    Polygon([(minx, miny), (midx, miny), (midx, midy), (minx, midy)]),
                    Polygon([(midx, miny), (maxx, miny), (maxx, midy), (midx, midy)]),
                    Polygon([(minx, midy), (midx, midy), (midx, maxy), (minx, maxy)]),
                    Polygon([(midx, midy), (maxx, midy), (maxx, maxy), (midx, maxy)])
                ]
                merged = MultiPolygon(polygons)
            else:
                merged = MultiPolygon([merged])

        return merged

    def save_geojson(self, multipoly, month):
        """保存GeoJSON文件，符合Copernicus要求"""
        # 创建GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=[multipoly], crs="EPSG:4326")

        # 文件路径
        filename = f"{month}_fire_density.geojson"
        output_path = os.path.join(self.output_dir, filename)

        # 手动构建GeoJSON以确保CRS格式正确
        geojson_dict = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": json.loads(gdf.geometry.to_json())['features'][0]['geometry']
            }],
            "crs": {
                "type": "name",
                "properties": {
                    "wkid": 4326
                }
            }
        }

        # 保存文件
        with open(output_path, 'w') as f:
            json.dump(geojson_dict, f, indent=2)

        # 检查文件大小
        file_size = os.path.getsize(output_path) / 1024  # KB
        tqdm.write(f"已保存 {filename} - 文件大小: {file_size:.2f} KB")

        if file_size > 10240:  # 10MB限制
            tqdm.write(f"警告: {filename} 大小超过10MB限制!")


if __name__ == "__main__":
    # 配置路径和参数
    INPUT_DIR = r"D:\Downloads\viirs-snpp_2015_all_countries\viirs-snpp\2015"  # 替换为实际路径
    OUTPUT_DIR = r"D:\Downloads\viirs-snpp_2015_all_countries\viirs-snpp\output"  # 替换为实际路径

    processor = FireDataProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        k=100,  # 每月选取的top-k密度区域
        grid_size=0.1  # 空间网格大小(度)
    )

    # 处理国家数据
    print("开始处理国家数据...")
    combined_df = processor.process_country_data()

    if combined_df is not None:
        # 处理月度数据
        print("\n开始处理月度数据...")
        processor.process_monthly_data(combined_df)

    print("\n处理完成!")
