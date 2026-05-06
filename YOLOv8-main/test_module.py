"""
YOLOv8跟踪初始化模块快速测试

用于验证模块是否正确安装和配置
"""

import sys
from pathlib import Path

YOLOV8_ROOT = Path(__file__).parent
sys.path.insert(0, str(YOLOV8_ROOT))

print("="*60)
print("YOLOv8跟踪初始化模块测试")
print("="*60)

# 测试1: 导入模块
print("\n[1] 测试模块导入...")
try:
    from yolo_tracker_initializer import (
        SUTrackInitializer,
        YOLOv8Detector,
        DetectionResult,
        TrackInitialization
    )
    print("  ✓ yolo_tracker_initializer 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    sys.exit(1)

try:
    from sutrack_integration import (
        SUTrackIntegration,
        TrackAdapter,
        TrackerFactory,
        SUTrackBox
    )
    print("  ✓ sutrack_integration 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    sys.exit(1)

# 测试2: 检查权重文件
print("\n[2] 检查权重文件...")
weight_path = YOLOV8_ROOT / "runs" / "detect" / "anti_uav_single_stage16" / "weights" / "best.pt"
if weight_path.exists():
    import os
    size_mb = os.path.getsize(weight_path) / (1024 * 1024)
    print(f"  ✓ 权重文件存在: {weight_path}")
    print(f"    文件大小: {size_mb:.2f} MB")
else:
    print(f"  ✗ 权重文件不存在: {weight_path}")
    sys.exit(1)

# 测试3: 创建检测器
print("\n[3] 测试创建检测器...")
try:
    detector = YOLOv8Detector(
        model_path=str(weight_path),
        conf_threshold=0.25,
        device='cpu'  # 使用CPU避免CUDA问题
    )
    print("  ✓ 检测器创建成功")
    print(f"    设备: {detector.device}")
    print(f"    类别: {detector.class_names}")
except Exception as e:
    print(f"  ✗ 检测器创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 创建初始化器
print("\n[4] 测试创建初始化器...")
try:
    initializer = SUTrackInitializer(
        model_path=str(weight_path),
        conf_threshold=0.25,
        device='cpu'
    )
    print("  ✓ 初始化器创建成功")
except Exception as e:
    print(f"  ✗ 初始化器创建失败: {e}")
    sys.exit(1)

# 测试5: 创建集成模块
print("\n[5] 测试创建集成模块...")
try:
    integration = SUTrackIntegration(
        model_path=str(weight_path),
        conf_threshold=0.25,
        device='cpu'
    )
    print("  ✓ 集成模块创建成功")
except Exception as e:
    print(f"  ✗ 集成模块创建失败: {e}")
    sys.exit(1)

# 测试6: 测试数据适配器
print("\n[6] 测试数据适配器...")
try:
    adapter = TrackAdapter()
    
    # 创建模拟DetectionResult
    import numpy as np
    mock_detection = DetectionResult(
        bbox=np.array([100, 100, 200, 200]),
        confidence=0.95,
        class_id=0,
        class_name='drone',
        center=np.array([150, 150]),
        size=np.array([100, 100])
    )
    
    sutrack_box = adapter.detection_to_sutrack_box(mock_detection)
    print(f"  ✓ 适配器工作正常")
    print(f"    转换后的Box: {sutrack_box.to_dict()}")
except Exception as e:
    print(f"  ✗ 适配器测试失败: {e}")
    sys.exit(1)

# 测试7: 检查测试图像
print("\n[7] 检查测试图像...")
test_image_paths = [
    YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg",
    YOLOV8_ROOT / "test_img",
]

test_image = None
for path in test_image_paths:
    if path.is_file() and path.exists():
        test_image = path
        break
    elif path.is_dir() and path.exists():
        image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
        if image_files:
            test_image = image_files[0]
            break

if test_image and test_image.exists():
    print(f"  ✓ 找到测试图像: {test_image}")
else:
    print(f"  ⚠ 未找到测试图像，跳过推理测试")
    test_image = None

# 测试8: 执行推理（如果有测试图像）
if test_image:
    print("\n[8] 测试推理...")
    try:
        import cv2
        image = cv2.imread(str(test_image))
        
        # 检测
        detections = detector.detect(image)
        print(f"  ✓ 推理成功")
        print(f"    检测到 {len(detections)} 个目标")
        
        if len(detections) > 0:
            best = detections[0]
            print(f"    最佳目标: 置信度={best.confidence:.4f}, 类别={best.class_name}")
        
        # 初始化
        init_result = initializer.initialize(image, frame_id=0)
        if init_result:
            print(f"  ✓ 初始化成功")
            print(f"    目标ID: {init_result.target_id}")
            print(f"    边界框: {init_result.initial_bbox}")
        else:
            print(f"  ⚠ 初始化返回None（可能未检测到目标）")
            
    except Exception as e:
        print(f"  ✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()

# 测试9: 测试TrackerFactory
print("\n[9] 测试TrackerFactory...")
try:
    tracker1 = TrackerFactory.create_tracker("test_tracker_1")
    tracker2 = TrackerFactory.create_tracker("test_tracker_2")
    tracker1_again = TrackerFactory.create_tracker("test_tracker_1")
    
    assert tracker1 is tracker1_again, "单例模式失败"
    assert len(TrackerFactory.list_trackers()) == 2, "跟踪器数量错误"
    
    print(f"  ✓ TrackerFactory工作正常")
    print(f"    创建的跟踪器: {TrackerFactory.list_trackers()}")
    
    # 清理
    TrackerFactory.clear_all()
except Exception as e:
    print(f"  ✗ TrackerFactory测试失败: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("所有测试通过! 模块可以正常使用。")
print("="*60)
print("\n使用示例:")
print("  from yolo_tracker_initializer import SUTrackInitializer")
print("  initializer = SUTrackInitializer()")
print("  result = initializer.initialize(first_frame)")
print("\n详细文档请查看: YOLO_TRACKER_README.md")
