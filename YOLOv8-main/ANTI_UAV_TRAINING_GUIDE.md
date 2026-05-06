# Anti-UAV 红外无人机检测 - YOLOv8 完整训练方案

## 项目概述

本项目基于 YOLOv8 实现高精度的红外无人机目标检测，针对 Anti-UAV 数据集的红外图像特点进行了专门的优化。

## 数据集分析

### Anti-UAV 数据集结构

```
data/AntI-UAV/
├── train/              # 训练集视频
│   ├── 20190925_140917_1_5/
│   │   ├── infrared.mp4    # 红外视频
│   │   └── visible.mp4     # 可见光视频
│   └── ...
├── val/                # 验证集视频
├── test/               # 测试集视频
├── label_new/          # 标签文件
│   ├── train.json      # 训练集属性标签
│   ├── val.json        # 验证集属性标签
│   └── test.json       # 测试集属性标签
└── framecut.py         # 视频转帧脚本
```

### 数据集属性说明

Anti-UAV 数据集标注了以下挑战属性：

| 属性 | 全称 | 说明 |
|------|------|------|
| FM | Fast Motion | 快速移动 |
| SV | Scale Variation | 尺度变化 |
| OV | Out-of-View | 目标离开视野 |
| TC | Thermal Crossover | 热交叉（红外特有）|
| TC-EASY | Thermal Crossover - Easy | 简单热交叉 |
| TC-MID | Thermal Crossover - Medium | 中等热交叉 |
| TC-HARD | Thermal Crossover - Hard | 困难热交叉 |
| LR | Low Resolution | 低分辨率 |
| LI | Light Illumination | 光照变化 |
| OC | Occlusion | 遮挡 |

## 一、数据集预处理

### 1.1 格式转换流程

```bash
# 1. 进入数据集目录
cd data/AntI-UAV

# 2. 运行预处理脚本
python prepare_dataset.py
```

预处理脚本将执行以下操作：

1. **视频转帧**：从红外视频中提取帧图像
   - 默认每5帧提取一张（可配置）
   - 输出到 `yolo_format/images/{train,val,test}/`

2. **标注转换**：将 groundtruth.txt 转换为 YOLO 格式
   - 原始格式：`x,y,w,h`（像素坐标）
   - YOLO格式：`class_id x_center y_center width height`（归一化）
   - 输出到 `yolo_format/labels/{train,val,test}/`

3. **生成配置文件**：创建 `anti_uav.yaml`

### 1.2 数据增强策略

针对红外图像特点，采用以下增强策略：

#### 基础增强（YOLOv8内置）

```yaml
# 颜色空间增强
hsv_h: 0.015          # 色调（红外图像不敏感，保持较小）
hsv_s: 0.3            # 饱和度（降低，红外图像饱和度低）
hsv_v: 0.5            # 亮度（提高，适应红外亮度变化）

# 几何变换
degrees: 5.0          # 旋转（无人机多角度）
translate: 0.1        # 平移
scale: 0.5            # 缩放（重要：无人机尺度变化大）
shear: 2.0            # 剪切

# 翻转
flipud: 0.0           # 上下翻转（无人机通常不翻转）
fliplr: 0.5           # 左右翻转

# 高级增强
mosaic: 1.0           # Mosaic增强（对小目标检测很有帮助）
mixup: 0.1            # MixUp增强
```

#### 红外特定增强（自定义实现）

```python
# 1. 红外亮度调整
ir_brightness: 0.3    # 模拟不同温度条件

# 2. 红外对比度调整  
ir_contrast: 0.3      # 增强目标与背景对比

# 3. 高斯噪声
ir_noise: 0.05        # 模拟红外传感器噪声

# 4. 随机模糊
ir_blur: 0.1          # 模拟失焦情况

# 5. 热交叉模拟
thermal_crossover: 0.2  # 模拟目标与背景热交叉
```

### 1.3 训练集与验证集划分

数据集已提供官方划分：

| 划分 | 视频数 | 用途 |
|------|--------|------|
| Train | ~60 | 模型训练 |
| Val | ~40 | 超参数调优、早停 |
| Test | ~20 | 最终性能评估 |

**注意**：保持官方划分，确保与论文结果可比。

## 二、模型配置

### 2.1 YOLOv8 版本选择

| 模型 | 参数量 | FLOPs | 适用场景 |
|------|--------|-------|----------|
| YOLOv8n | 3.2M | 8.7B | 边缘设备、实时应用 |
| **YOLOv8s** | **11.2M** | **28.6B** | **推荐：平衡精度与速度** |
| YOLOv8m | 25.9M | 78.9B | 高精度需求 |
| YOLOv8l | 43.7M | 165.2B | 高精度、服务器部署 |
| YOLOv8x | 68.2M | 257.8B | 最高精度 |

**推荐**：YOLOv8s，在红外小目标检测上有较好的性价比。

### 2.2 网络结构调整

针对红外无人机检测的特点，在 `configs/anti_uav_model.yaml` 中优化：

```yaml
# 类别数
nc: 1  # 只有无人机一个类别

# 小目标检测优化
# P3层（8倍下采样）专门针对小目标
# 保留更多浅层特征信息

# 损失函数权重调整
box: 7.5    # 提高边框回归权重
cls: 0.5    # 分类权重（单类别可适当降低）
dfl: 1.5    # 分布焦点损失
```

### 2.3 超参数设置

```yaml
# 训练参数
epochs: 200           # 红外数据集需要更多轮数
patience: 30          # 早停耐心值
batch: 16             # 根据GPU显存调整（8GB建议8-16）
imgsz: 640            # 输入尺寸

# 优化器
optimizer: SGD        # SGD在检测任务上通常更好
lr0: 0.01             # 初始学习率
lrf: 0.01             # 最终学习率
momentum: 0.937       # 动量
weight_decay: 0.0005  # 权重衰减

# 学习率调度
cos_lr: True          # 余弦退火
warmup_epochs: 3.0    # 预热轮数
```

## 三、训练流程

### 3.1 分阶段训练策略

#### 方案一：分阶段训练（推荐）

**阶段1：冻结骨干网络训练（50轮）**

```bash
python train_anti_uav.py --mode phased --model yolov8s.pt
```

- 冻结前10层（Backbone）
- 只训练检测头（Head）
- 学习率：0.001（较低）
- 目的：在预训练特征基础上快速适应红外数据

**阶段2：全网络微调（150轮）**

- 解冻所有层
- 端到端训练
- 学习率：0.01
- 更强的数据增强
- 目的：精细调整整个网络

#### 方案二：单阶段训练

```bash
python train_anti_uav.py --mode single --model yolov8s.pt --epochs 200
```

适用于：
- 快速实验
- 计算资源充足
- 从头训练

### 3.2 训练监控指标

#### 损失函数

| 损失类型 | 说明 | 正常范围 |
|----------|------|----------|
| box_loss | 边框回归损失（CIoU） | 0.5-2.0 |
| cls_loss | 分类损失（VFL） | 0.1-1.0 |
| dfl_loss | 分布焦点损失 | 0.5-2.0 |

#### 验证指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| mAP@0.5 | IoU=0.5时的平均精度 | > 0.85 |
| mAP@0.5:0.95 | COCO标准mAP | > 0.60 |
| Precision | 精确率 | > 0.85 |
| Recall | 召回率 | > 0.80 |

### 3.3 模型保存策略

```yaml
save: True            # 启用保存
save_period: 10       # 每10轮保存一次
checkpoint: True      # 保存检查点

# 自动保存
# - best.pt: 验证集上最佳模型
# - last.pt: 最后一轮模型
# - epoch{xxx}.pt: 每save_period轮保存
```

### 3.4 训练命令示例

```bash
# 1. 分阶段训练（推荐）
python train_anti_uav.py --mode phased --model yolov8s.pt --batch 16 --device 0

# 2. 单阶段训练
python train_anti_uav.py --mode single --model yolov8s.pt --epochs 200 --batch 16

# 3. 恢复训练
python train_anti_uav.py --mode resume --model runs/detect/anti_uav/last.pt

# 4. 训练并验证
python train_anti_uav.py --mode single --model yolov8s.pt --validate --export
```

## 四、模型评估

### 4.1 评估指标

#### 主要指标

| 指标 | 计算公式 | 说明 |
|------|----------|------|
| **mAP@0.5** | 平均精度（IoU=0.5） | 主要指标，衡量检测准确性 |
| **mAP@0.5:0.95** | COCO标准mAP | 综合指标，考虑不同IoU阈值 |
| **Precision** | TP / (TP + FP) | 精确率，减少误检 |
| **Recall** | TP / (TP + FN) | 召回率，减少漏检 |
| **F1-Score** | 2 * P * R / (P + R) | 精确率和召回率的调和平均 |

#### 红外特定指标

- **TC场景mAP**：在热交叉场景下的性能
- **小目标mAP**：目标面积<32x32像素的性能
- **夜间mAP**：夜间场景的性能

### 4.2 评估方法

```bash
# 基础评估
python evaluate_anti_uav.py --model runs/detect/anti_uav/weights/best.pt --data data/AntI-UAV/yolo_format/anti_uav.yaml

# 完整评估（包含所有分析）
python evaluate_anti_uav.py \
    --model runs/detect/anti_uav/weights/best.pt \
    --data data/AntI-UAV/yolo_format/anti_uav.yaml \
    --splits val test \
    --multi-conf \
    --speed \
    --attribute \
    --report
```

### 4.3 评估报告格式

```
============================================================
Anti-UAV 红外无人机检测模型评估报告
============================================================
模型: runs/detect/anti_uav/weights/best.pt
数据集: data/AntI-UAV/yolo_format/anti_uav.yaml

【VAL 集评估结果】
----------------------------------------
mAP@0.5:      0.8923
mAP@0.5:0.95: 0.6543
mAP@0.75:     0.7234
Precision:    0.9123
Recall:       0.8567
F1-Score:     0.8834

【TEST 集评估结果】
----------------------------------------
mAP@0.5:      0.8745
mAP@0.5:0.95: 0.6234
...

【推理速度】
平均推理时间: 12.34 ± 1.23 ms
FPS: 81.05

【属性分析】
TC-HARD场景: 50个视频
SV场景: 80个视频
...
============================================================
```

## 五、快速开始

### 5.1 环境准备

```bash
# 1. 安装依赖
pip install ultralytics==8.0.0
pip install opencv-python numpy matplotlib tqdm

# 2. 验证安装
python -c "from ultralytics import YOLO; print('OK')"
```

### 5.2 数据准备

```bash
cd data/AntI-UAV
python prepare_dataset.py
```

### 5.3 开始训练

```bash
# 分阶段训练（推荐）
python train_anti_uav.py --mode phased --model yolov8s.pt

# 或单阶段训练
python train_anti_uav.py --mode single --model yolov8s.pt --epochs 200
```

### 5.4 模型评估

```bash
python evaluate_anti_uav.py \
    --model runs/detect/anti_uav_phase2/weights/best.pt \
    --data data/AntI-UAV/yolo_format/anti_uav.yaml \
    --splits val test \
    --report
```

## 六、注意事项

### 6.1 红外图像特点

1. **低对比度**：目标与背景温差小
2. **热交叉**：目标与背景温度相近时难以区分
3. **噪声大**：红外传感器固有噪声
4. **小目标**：无人机距离远时成像小

### 6.2 训练技巧

1. **数据增强**：Mosaic增强对小目标检测非常有帮助
2. **学习率**：红外数据集建议使用较小的初始学习率
3. **训练轮数**：红外数据集通常需要更多轮数收敛
4. **多尺度训练**：有助于处理尺度变化

### 6.3 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| mAP上不去 | 学习率过大/数据增强不足 | 降低lr0，增强mosaic |
| 召回率低 | 置信度阈值过高 | 降低conf阈值 |
| 小目标检测差 | 特征图分辨率不足 | 增大输入尺寸或使用P2层 |
| 过拟合 | 数据量不足/训练轮数过多 | 增加数据增强，使用早停 |

## 七、文件说明

| 文件 | 说明 |
|------|------|
| `data/AntI-UAV/prepare_dataset.py` | 数据集预处理脚本 |
| `configs/anti_uav_training.yaml` | 训练配置文件 |
| `configs/anti_uav_model.yaml` | 模型结构配置文件 |
| `train_anti_uav.py` | 训练启动脚本 |
| `evaluate_anti_uav.py` | 模型评估脚本 |

## 八、参考

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [Anti-UAV数据集论文](https://github.com/ZhaoJ9014/Anti-UAV)
- [YOLOv8论文](https://arxiv.org/abs/2301.05586)
