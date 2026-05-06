"""测试修复后的YOLOv8Detector类"""
import sys
from pathlib import Path

YOLOV8_ROOT = Path(__file__).parent
sys.path.insert(0, str(YOLOV8_ROOT))

print("="*60)
print("测试YOLOv8Detector修复")
print("="*60)

try:
    print("\n[1] 导入模块...")
    from yolo_tracker_initializer import YOLOv8Detector
    print("  ✓ 导入成功")
    
    print("\n[2] 创建检测器...")
    detector = YOLOv8Detector(
        model_path=None,  # 使用默认路径
        conf_threshold=0.25,
        device='cpu'
    )
    print("  ✓ 检测器创建成功")
    print(f"  ✓ 类别名称: {detector.class_names}")
    
    print("\n[3] 测试检测功能...")
    import cv2
    
    # 查找测试图像
    test_image = None
    possible_paths = [
        YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg",
        YOLOV8_ROOT / "test_img",
    ]
    
    for path in possible_paths:
        if path.is_file() and path.exists():
            test_image = path
            break
        elif path.is_dir() and path.exists():
            image_files = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if image_files:
                test_image = image_files[0]
                break
    
    if test_image and test_image.exists():
        print(f"  使用测试图像: {test_image}")
        image = cv2.imread(str(test_image))
        
        print("  执行检测...")
        detections = detector.detect(image)
        print(f"  ✓ 检测到 {len(detections)} 个目标")
        
        if len(detections) > 0:
            best = detections[0]
            print(f"  ✓ 最佳目标: 置信度={best.confidence:.4f}, 类别={best.class_name}")
    else:
        print("  ⚠ 未找到测试图像，跳过检测测试")
    
    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
