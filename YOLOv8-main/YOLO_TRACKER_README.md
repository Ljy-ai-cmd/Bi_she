# YOLOv8红外目标检测与SUTrack跟踪初始化模块

## 模块概述

本模块实现基于YOLOv8的红外无人机目标检测，并为SUTrack跟踪系统提供第一帧初始化功能。

### 主要功能

1. **目标检测**：使用YOLOv8模型对红外图像进行目标检测
2. **单目标筛选**：自动选择置信度最高的单个无人机目标
3. **跟踪初始化**：基于检测结果初始化SUTrack跟踪器
4. **数据格式转换**：提供SUTrack系统兼容的数据格式
5. **状态管理**：管理跟踪目标的状态和历史轨迹

## 文件结构

```
YOLOv8-main/
├── yolo_tracker_initializer.py    # 核心初始化模块
├── sutrack_integration.py         # SUTrack系统集成模块
├── example_usage.py               # 使用示例
├── YOLO_TRACKER_README.md         # 本文档
└── runs/detect/anti_uav_single_stage16/weights/best.pt  # 默认权重
```

## 安装依赖

确保已安装YOLOv8所需的依赖：

```bash
pip install ultralytics opencv-python numpy torch
```

## 快速开始

### 1. 基本使用

```python
from yolo_tracker_initializer import SUTrackInitializer
import cv2

# 创建初始化器
initializer = SUTrackInitializer()

# 读取第一帧图像
first_frame = cv2.imread("path/to/first_frame.jpg")

# 初始化跟踪
init_result = initializer.initialize(first_frame, frame_id=0)

if init_result:
    print(f"目标ID: {init_result.target_id}")
    print(f"初始边界框: {init_result.initial_bbox}")
    print(f"置信度: {init_result.confidence}")
```

### 2. SUTrack系统集成

```python
from sutrack_integration import SUTrackIntegration
import cv2

# 创建集成模块
integration = SUTrackIntegration()

# 初始化并获取SUTrack格式数据
first_frame = cv2.imread("path/to/first_frame.jpg")
sutrack_data = integration.initialize_first_frame(first_frame)

# sutrack_data包含SUTrack系统所需的所有信息
print(f"初始边界框: {sutrack_data['init_bbox']}")
print(f"中心点: {sutrack_data['init_center']}")
print(f"尺寸: {sutrack_data['init_size']}")
```

### 3. 快速初始化（一行代码）

```python
from sutrack_integration import quick_initialize
import cv2

first_frame = cv2.imread("path/to/first_frame.jpg")
result = quick_initialize(first_frame)

print(f"初始化结果: {result}")
```

## 详细API说明

### yolo_tracker_initializer.py

#### DetectionResult

检测结果数据结构：

```python
@dataclass
class DetectionResult:
    bbox: np.ndarray          # 边界框 [x1, y1, x2, y2]
    confidence: float         # 置信度 (0-1)
    class_id: int            # 类别ID
    class_name: str          # 类别名称
    center: np.ndarray       # 中心点 [cx, cy]
    size: np.ndarray         # 尺寸 [width, height]
```

#### TrackInitialization

跟踪初始化数据结构：

```python
@dataclass
class TrackInitialization:
    target_id: int                    # 目标唯一标识
    initial_bbox: np.ndarray          # 初始边界框
    initial_center: np.ndarray        # 初始中心点
    initial_size: np.ndarray          # 初始尺寸
    confidence: float                 # 检测置信度
    frame_id: int                     # 初始化帧ID
    timestamp: float                  # 时间戳
    metadata: Dict                    # 元数据
```

#### YOLOv8Detector

YOLOv8检测器类：

```python
detector = YOLOv8Detector(
    model_path=None,           # 模型路径，None使用默认路径
    conf_threshold=0.25,       # 置信度阈值
    iou_threshold=0.45,        # NMS IoU阈值
    device='auto'              # 计算设备 ('cpu', 'cuda', 'auto')
)

# 检测所有目标
detections = detector.detect(image)

# 检测单目标（置信度最高）
best_detection = detector.detect_single_target(image)
```

#### SUTrackInitializer

SUTrack初始化器类：

```python
initializer = SUTrackInitializer(
    model_path=None,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device='auto'
)

# 初始化跟踪
init_result = initializer.initialize(first_frame, frame_id=0)

# 可视化
vis_image = initializer.visualize_detection(
    image, 
    init_result, 
    save_path="result.jpg"
)
```

### sutrack_integration.py

#### SUTrackIntegration

SUTrack系统集成类：

```python
integration = SUTrackIntegration(
    model_path=None,
    conf_threshold=0.25,
    iou_threshold=0.45,
    device='auto'
)

# 初始化并获取SUTrack格式数据
sutrack_data = integration.initialize_first_frame(first_frame)

# 获取初始化信息
info = integration.get_initialization_info()
```

#### TrackAdapter

数据格式适配器：

```python
from sutrack_integration import TrackAdapter

adapter = TrackAdapter()

# 转换为SUTrackBox
sutrack_box = adapter.detection_to_sutrack_box(detection)

# 转换为完整SUTrack格式
sutrack_data = adapter.to_sutrack_format(initialization)
```

#### TrackerFactory

跟踪器工厂（单例模式）：

```python
from sutrack_integration import TrackerFactory

# 创建跟踪器
tracker = TrackerFactory.create_tracker("my_tracker")

# 获取已存在的跟踪器
tracker = TrackerFactory.get_tracker("my_tracker")

# 列出所有跟踪器
tracker_ids = TrackerFactory.list_trackers()

# 移除跟踪器
TrackerFactory.remove_tracker("my_tracker")

# 清除所有
TrackerFactory.clear_all()
```

## 配置参数

### 检测参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | None | 模型权重路径，None使用默认路径 |
| `conf_threshold` | float | 0.25 | 置信度阈值，低于此值的目标被过滤 |
| `iou_threshold` | float | 0.45 | NMS IoU阈值，用于去除重叠框 |
| `device` | str | 'auto' | 计算设备，'cpu'、'cuda'或'auto' |

### 默认权重路径

```
YOLOv8-main/runs/detect/anti_uav_single_stage16/weights/best.pt
```

## 数据格式

### SUTrack格式输出

```python
{
    'target_id': 1,
    'init_bbox': [x1, y1, x2, y2],        # 边界框坐标
    'init_center': [cx, cy],              # 中心点坐标
    'init_size': [width, height],         # 目标尺寸
    'confidence': 0.95,                   # 置信度
    'frame_id': 0,                        # 帧ID
    'box_dict': {                         # 详细的边界框信息
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
        'cx': cx, 'cy': cy,
        'w': width, 'h': height,
        'confidence': 0.95
    },
    'metadata': {                         # 元数据
        'class_id': 0,
        'class_name': 'drone',
        'detection_method': 'YOLOv8',
        'model_path': '...'
    }
}
```

## 使用示例

### 示例1：基本检测

```python
from yolo_tracker_initializer import YOLOv8Detector
import cv2

detector = YOLOv8Detector(conf_threshold=0.25)
image = cv2.imread("infrared_image.jpg")
detections = detector.detect(image)

for det in detections:
    print(f"检测到 {det.class_name}, 置信度: {det.confidence:.4f}")
```

### 示例2：视频序列初始化

```python
from sutrack_integration import SUTrackIntegration
import cv2

integration = SUTrackIntegration()

# 打开视频
cap = cv2.VideoCapture("video.mp4")
ret, first_frame = cap.read()

if ret:
    # 初始化
    init_data = integration.initialize_first_frame(first_frame)
    
    # 传递给SUTrack跟踪器
    # tracker = SUTrackTracker(init_data)
    
    # 后续帧跟踪
    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     tracker.update(frame)
```

### 示例3：批量处理

```python
from sutrack_integration import batch_initialize

image_paths = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
results = batch_initialize(image_paths)

for i, result in enumerate(results):
    if result:
        print(f"图像 {i+1}: 初始化成功")
    else:
        print(f"图像 {i+1}: 初始化失败")
```

## 测试

运行模块自带的测试：

```bash
# 测试核心模块
python yolo_tracker_initializer.py

# 测试集成模块
python sutrack_integration.py

# 运行使用示例
python example_usage.py
```

## 注意事项

1. **权重文件**：确保权重文件 `best.pt` 存在于指定路径
2. **GPU内存**：如果使用GPU，确保有足够的显存
3. **输入图像**：支持BGR格式的OpenCV图像
4. **置信度阈值**：根据实际场景调整，红外无人机建议0.25-0.5
5. **单目标假设**：当前实现假设每帧只有一个目标，取置信度最高的

## 故障排除

### 问题1：模型加载失败

```
FileNotFoundError: 模型权重文件不存在
```

**解决方案**：检查权重文件路径是否正确，或指定正确的 `model_path`

### 问题2：CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：设置 `device='cpu'` 使用CPU推理

### 问题3：未检测到目标

```
警告: 未检测到任何目标
```

**解决方案**：降低 `conf_threshold` 阈值，或检查图像质量

## 性能优化

1. **使用GPU**：设置 `device='cuda'` 加速推理
2. **批量处理**：对多帧图像使用批量处理接口
3. **调整输入尺寸**：根据需要调整模型输入尺寸
4. **降低精度**：使用FP16半精度推理（需要模型支持）

## 更新日志

- **2026-04-20**: 初始版本发布
  - 实现YOLOv8检测器
  - 实现SUTrack初始化器
  - 实现系统集成模块
  - 提供完整API文档和示例

## 作者

毕业设计项目 - 基于单目相机的红外目标定位方法研究

## 许可证

本项目仅供学术研究使用
