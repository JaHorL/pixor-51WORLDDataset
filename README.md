# PIXOR: Real-time 3D Object Detection from Point Clouds
## 简介

这是PIXOR算法的非官方实现，可以直接使用51Sim-One数据集进行训练和测试。

在这个版本的实现中，主要有两点不同: 1）点云数据的预处理方法，这边借鉴complex-yolo的点云预处理方法；2）网络结构更加轻量级。

## 环境配置

```
opencv
path
tensorlfow >= 1.14
easydict
numpy
python
```

## 训练与测试

1. 下载数据集，并解压至以下目录结构

```
├── train
    ├── scene0
        ├── DumpSettings.json
        ├── pcd_bin
        ├── pcd_label
    ...
├── test
    ├── scene1
        ├── DumpSettings.json
        ├── pcd_bin
    ...
```

2. 运行训练脚本

```
python train.py
```

3. 运行测试脚本

```
python predict.py 
```

## Credit

PIXOR: Real-time 3D Object Detection from Point Clouds