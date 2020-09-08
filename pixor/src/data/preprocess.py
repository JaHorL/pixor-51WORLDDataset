import numpy as np
from config.config import cfg
# from data.lidar_preprocess.build import libcuda_preprocessor
from utils import transform


def lidar_preprocess(point_cloud):
    bev = convert_points_to_bev(point_cloud)    
    return bev


def removePoints(PointCloud):
    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= cfg.BEV.X_MIN) & (PointCloud[:, 0] <= cfg.BEV.X_MAX) & (PointCloud[:, 1] >= cfg.BEV.Y_MIN) & (
            PointCloud[:, 1] <= cfg.BEV.Y_MAX) & (PointCloud[:, 2] >= cfg.BEV.Z_MIN) & (PointCloud[:, 2] <= cfg.BEV.Z_MAX))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] - cfg.BEV.Z_MIN

    return PointCloud


def convert_points_to_bev(point_cloud):
    point_cloud = removePoints(point_cloud)
    height = int(np.abs(cfg.BEV.X_MAX-cfg.BEV.X_MIN) / cfg.BEV.X_RESOLUTION + 1)
    width = int(np.abs(cfg.BEV.Y_MAX-cfg.BEV.Y_MIN) / cfg.BEV.Y_RESOLUTION + 1)
    # print(height, width)
    # Discretize Feature Map
    points = np.copy(point_cloud)
    points[:, 0] = -np.int_(np.floor(points[:, 0] / cfg.BEV.X_RESOLUTION))
    points[:, 1] = -np.int_(np.floor(points[:, 1] / cfg.BEV.Y_RESOLUTION) + width / 2)

    # sort-3times
    indices = np.lexsort((-points[:, 2], points[:, 1], points[:, 0]))
    points = points[indices]

    # Height Map
    heightMap = np.zeros((height, width))

    _, indices = np.unique(points[:, 0:2], axis=0, return_index=True)
    points_frac = points[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(cfg.BEV.Z_MAX-cfg.BEV.Z_MIN))
    heightMap[np.int_(points_frac[:, 0]), np.int_(points_frac[:, 1])] = points_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((height, width))
    densityMap = np.zeros((height, width))

    _, indices, counts = np.unique(points[:, 0:2], axis=0, return_index=True, return_counts=True)
    points_top = points[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(points_top[:, 0]), np.int_(points_top[:, 1])] = points_top[:, 3]
    densityMap[np.int_(points_top[:, 0]), np.int_(points_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((height - 1, width - 1, 3))
    RGB_Map[..., 2] = densityMap[:height - 1, :width-1]  # r_map
    RGB_Map[..., 1] = heightMap[:height - 1, :width-1]  # g_map
    RGB_Map[..., 0] = intensityMap[:height - 1, :width-1]  # b_map
    return RGB_Map