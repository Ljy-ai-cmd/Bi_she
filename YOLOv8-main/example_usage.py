"""
YOLOv8红外目标检测与SUTrack集成使用示例

本示例展示如何在SUTrack系统中使用YOLOv8进行第一帧目标检测和跟踪初始化
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加YOLOv8路径
YOLOV8_ROOT = Path(__file__).parent
sys.path.insert(0, str(YOLOV8_ROOT))

from yolo_tracker_initializer import (
    SUTrackInitializer,
    YOLOv8Detector,
    initialize_tracking
)
from sutrack_integration import (
    SUTrackIntegration,
    quick_initialize,
    TrackerFactory
)


def example_1_basic_detection():
    """示例1: 基本目标检测"""
    print("\n" + "="*60)
    print("示例1: 基本目标检测")
    print("="*60)
    
    # 创建检测器
    detector = YOLOv8Detector(
        model_path=None,  # 使用默认路径
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # 测试图像路径
    test_image = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    if not test_image.exists():
        print(f"测试图像不存在: {test_image}")
        return
    
    # 读取图像
    image = cv2.imread(str(test_image))
    
    # 执行检测
    detections = detector.detect(image)
    
    print(f"检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections[:3]):
        print(f"  目标 {i+1}:")
        print(f"    类别: {det.class_name}")
        print(f"    置信度: {det.confidence:.4f}")
        print(f"    边界框: [{det.bbox[0]:.1f}, {det.bbox[1]:.1f}, {det.bbox[2]:.1f}, {det.bbox[3]:.1f}]")
        print(f"    中心点: ({det.center[0]:.1f}, {det.center[1]:.1f})")
    
    # 单目标检测
    best_detection = detector.detect_single_target(image)
    if best_detection:
        print(f"\n最佳目标:")
        print(f"  置信度: {best_detection.confidence:.4f}")
        print(f"  边界框: {best_detection.bbox}")


def example_2_track_initialization():
    """示例2: 跟踪初始化"""
    print("\n" + "="*60)
    print("示例2: 跟踪初始化")
    print("="*60)
    
    # 创建初始化器
    initializer = SUTrackInitializer(
        conf_threshold=0.25,
        device='auto'
    )
    
    # 测试图像
    test_image = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    if not test_image.exists():
        print(f"测试图像不存在: {test_image}")
        return
    
    # 读取图像
    image = cv2.imread(str(test_image))
    
    # 初始化跟踪
    init_result = initializer.initialize(image, frame_id=0)
    
    if init_result is None:
        print("初始化失败!")
        return
    
    print("初始化成功!")
    print(f"  目标ID: {init_result.target_id}")
    print(f"  初始边界框: {init_result.initial_bbox}")
    print(f"  初始中心点: {init_result.initial_center}")
    print(f"  初始尺寸: {init_result.initial_size}")
    print(f"  置信度: {init_result.confidence:.4f}")
    print(f"  帧ID: {init_result.frame_id}")
    print(f"  元数据: {init_result.metadata}")
    
    # 可视化
    output_path = YOLOV8_ROOT / "test_results" / "example_2_result.jpg"
    output_path.parent.mkdir(exist_ok=True)
    
    vis_image = initializer.visualize_detection(image, init_result, str(output_path))
    print(f"\n可视化结果已保存: {output_path}")


def example_3_sutrack_integration():
    """示例3: SUTrack系统集成"""
    print("\n" + "="*60)
    print("示例3: SUTrack系统集成")
    print("="*60)
    
    # 创建集成模块
    integration = SUTrackIntegration(
        conf_threshold=0.25,
        device='auto'
    )
    
    # 测试图像
    test_image = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    if not test_image.exists():
        print(f"测试图像不存在: {test_image}")
        return
    
    # 读取图像
    image = cv2.imread(str(test_image))
    
    # SUTrack格式初始化
    sutrack_data = integration.initialize_first_frame(image, frame_id=0)
    
    if sutrack_data is None:
        print("初始化失败!")
        return
    
    print("SUTrack集成初始化成功!")
    print(f"  目标ID: {sutrack_data['target_id']}")
    print(f"  初始边界框 (xyxy): {sutrack_data['init_bbox']}")
    print(f"  初始中心点: {sutrack_data['init_center']}")
    print(f"  初始尺寸: {sutrack_data['init_size']}")
    print(f"  置信度: {sutrack_data['confidence']:.4f}")
    
    # Box字典包含更多格式
    box_dict = sutrack_data['box_dict']
    print(f"\n边界框详细信息:")
    print(f"  左上角 (x1, y1): ({box_dict['x1']:.1f}, {box_dict['y1']:.1f})")
    print(f"  右下角 (x2, y2): ({box_dict['x2']:.1f}, {box_dict['y2']:.1f})")
    print(f"  中心点 (cx, cy): ({box_dict['cx']:.1f}, {box_dict['cy']:.1f})")
    print(f"  尺寸 (w, h): ({box_dict['w']:.1f}, {box_dict['h']:.1f})")
    
    # 获取初始化信息
    info = integration.get_initialization_info()
    print(f"\n模型信息:")
    print(f"  模型路径: {info['model_path']}")
    print(f"  类别: {info['class_names']}")
    print(f"  设备: {info['device']}")


def example_4_quick_initialize():
    """示例4: 快速初始化"""
    print("\n" + "="*60)
    print("示例4: 快速初始化")
    print("="*60)
    
    # 测试图像
    test_image = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    if not test_image.exists():
        print(f"测试图像不存在: {test_image}")
        return
    
    # 读取图像
    image = cv2.imread(str(test_image))
    
    # 快速初始化 (一行代码)
    result = quick_initialize(
        image,
        conf_threshold=0.25,
        return_format='sutrack'
    )
    
    if result:
        print("快速初始化成功!")
        print(f"  目标ID: {result['target_id']}")
        print(f"  边界框: {result['init_bbox']}")
        print(f"  置信度: {result['confidence']:.4f}")


def example_5_tracker_factory():
    """示例5: 跟踪器工厂"""
    print("\n" + "="*60)
    print("示例5: 跟踪器工厂")
    print("="*60)
    
    # 创建多个跟踪器
    tracker_1 = TrackerFactory.create_tracker("uav_tracker_1")
    tracker_2 = TrackerFactory.create_tracker("uav_tracker_2")
    tracker_1_again = TrackerFactory.create_tracker("uav_tracker_1")  # 返回已存在的
    
    print(f"创建的跟踪器: {TrackerFactory.list_trackers()}")
    print(f"tracker_1 是同一个实例: {tracker_1 is tracker_1_again}")
    
    # 获取跟踪器
    retrieved_tracker = TrackerFactory.get_tracker("uav_tracker_1")
    print(f"获取跟踪器成功: {retrieved_tracker is not None}")
    
    # 移除跟踪器
    TrackerFactory.remove_tracker("uav_tracker_2")
    print(f"移除后剩余: {TrackerFactory.list_trackers()}")
    
    # 清理所有
    TrackerFactory.clear_all()
    print(f"清理后: {TrackerFactory.list_trackers()}")


def example_6_video_sequence():
    """示例6: 视频序列处理"""
    print("\n" + "="*60)
    print("示例6: 视频序列处理 (仅第一帧初始化)")
    print("="*60)
    
    # 创建集成模块
    integration = SUTrackIntegration(conf_threshold=0.25)
    
    # 测试图像 (模拟视频第一帧)
    test_image = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    if not test_image.exists():
        print(f"测试图像不存在: {test_image}")
        return
    
    # 模拟视频处理流程
    print("模拟视频跟踪流程:")
    print("  1. 读取第一帧...")
    first_frame = cv2.imread(str(test_image))
    
    print("  2. YOLOv8检测并初始化...")
    init_data = integration.initialize_first_frame(first_frame, frame_id=0)
    
    if init_data is None:
        print("  初始化失败!")
        return
    
    print(f"  3. 初始化成功!")
    print(f"     - 目标ID: {init_data['target_id']}")
    print(f"     - 初始位置: [{init_data['init_bbox'][0]:.1f}, {init_data['init_bbox'][1]:.1f}, "
          f"{init_data['init_bbox'][2]:.1f}, {init_data['init_bbox'][3]:.1f}]")
    print(f"     - 置信度: {init_data['confidence']:.4f}")
    
    print("  4. 将初始化数据传递给SUTrack跟踪器...")
    print("  5. 开始后续帧跟踪...")
    
    # 保存初始化结果
    output_dir = YOLOV8_ROOT / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    # 可视化
    vis_image = integration.initializer.visualize_detection(
        first_frame,
        integration.initializer.initialize(first_frame),
        str(output_dir / "video_init_result.jpg")
    )
    
    print(f"\n  初始化可视化已保存: {output_dir / 'video_init_result.jpg'}")


def example_7_custom_parameters():
    """示例7: 自定义参数"""
    print("\n" + "="*60)
    print("示例7: 自定义检测参数")
    print("="*60)
    
    # 测试图像
    test_image = YOLOV8_ROOT / "data" / "AntI-UAV" / "yolo_subset" / "images" / "test" / "20190926_134054_1_9_frame000030.jpg"
    
    if not test_image.exists():
        print(f"测试图像不存在: {test_image}")
        return
    
    image = cv2.imread(str(test_image))
    
    # 不同置信度阈值对比
    thresholds = [0.1, 0.25, 0.5, 0.7]
    
    print("不同置信度阈值的效果:")
    for conf_thresh in thresholds:
        detector = YOLOv8Detector(conf_threshold=conf_thresh)
        detections = detector.detect(image)
        print(f"  阈值 {conf_thresh}: 检测到 {len(detections)} 个目标")


def main():
    """主函数 - 运行所有示例"""
    print("="*60)
    print("YOLOv8红外目标检测与SUTrack集成使用示例")
    print("="*60)
    
    try:
        # 运行所有示例
        example_1_basic_detection()
        example_2_track_initialization()
        example_3_sutrack_integration()
        example_4_quick_initialize()
        example_5_tracker_factory()
        example_6_video_sequence()
        example_7_custom_parameters()
        
        print("\n" + "="*60)
        print("所有示例运行完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
