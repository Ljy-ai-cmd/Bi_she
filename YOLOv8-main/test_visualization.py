"""
使用训练好的模型可视化测试指定图片
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


def visualize_detection(model_path, image_path, conf=0.25, save=True, show=True):
    """
    可视化单张图片的检测结果
    
    Args:
        model_path: 训练好的模型路径
        image_path: 测试图片路径
        conf: 置信度阈值
        save: 是否保存结果
        show: 是否显示结果
    """
    # 加载模型
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 读取图片
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"错误: 图片不存在 {image_path}")
        return
    
    print(f"测试图片: {image_path}")
    
    # 进行检测
    results = model(image_path, conf=conf, verbose=True)
    
    # 获取结果 - YOLOv8 8.0.0 返回的是 Results 对象
    result = results[0]
    
    # 显示检测信息
    print(f"\n检测结果:")
    boxes = result.boxes
    if boxes is not None:
        print(f"  检测到 {len(boxes)} 个目标")
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            print(f"  目标 {i+1}: 类别={cls_id}, 置信度={conf_score:.3f}, 坐标={xyxy}")
    else:
        print(f"  未检测到目标")
    
    # 获取绘制了检测框的图片
    annotated_frame = result.plot()
    
    # 保存结果
    if save:
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"result_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"\n结果已保存: {output_path}")
    
    # 显示结果
    if show:
        # 转换 BGR 到 RGB 用于 matplotlib
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_frame_rgb)
        plt.title(f"Detection Result - {image_path.name}")
        plt.axis('off')
        plt.tight_layout()
        
        # 保存 matplotlib 图片
        if save:
            output_path_plt = output_dir / f"result_{image_path.stem}_plt.png"
            plt.savefig(output_path_plt, dpi=150, bbox_inches='tight')
            print(f"Matplotlib结果已保存: {output_path_plt}")
        
        plt.show()
    
    return results


def batch_test(model_path, image_dir, conf=0.25, save=True):
    """
    批量测试目录中的所有图片
    
    Args:
        model_path: 训练好的模型路径
        image_dir: 图片目录
        conf: 置信度阈值
        save: 是否保存结果
    """
    model = YOLO(model_path)
    image_dir = Path(image_dir)
    
    # 获取所有图片
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    print(f"找到 {len(image_files)} 张图片")
    
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    for img_path in image_files:
        print(f"\n处理: {img_path.name}")
        results = model(img_path, conf=conf, verbose=False)
        result = results[0]
        
        # 保存结果
        if save:
            annotated_frame = result.plot()
            output_path = output_dir / f"result_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_frame)
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"  检测到 {num_detections} 个目标 -> 保存到 {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 红外无人机检测可视化测试')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/anti_uav_single_stage16/weights/best.pt',
                       help='训练好的模型路径')
    parser.add_argument('--image', type=str, 
                       default=None,
                       help='单张测试图片路径')
    parser.add_argument('--dir', type=str, 
                       default=None,
                       help='批量测试图片目录')
    parser.add_argument('--conf', type=float, 
                       default=0.25,
                       help='置信度阈值 (默认0.25)')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示结果，只保存')
    
    args = parser.parse_args()
    
    # 检查模型是否存在
    if not Path(args.model).exists():
        print(f"错误: 模型不存在 {args.model}")
        print("可用的模型:")
        for model_path in Path("runs/detect").glob("*/weights/best.pt"):
            print(f"  - {model_path}")
        exit(1)
    
    show = not args.no_show
    
    if args.image:
        # 单张图片测试
        visualize_detection(args.model, args.image, args.conf, save=True, show=show)
    elif args.dir:
        # 批量测试
        batch_test(args.model, args.dir, args.conf, save=True)
    else:
        # 默认测试数据集中的图片
        test_images = list(Path("data/AntI-UAV/yolo_subset/images/test").glob("*.jpg"))
        if test_images:
            print(f"使用测试集中的图片: {test_images[0]}")
            visualize_detection(args.model, test_images[0], args.conf, save=True, show=show)
        else:
            print("请提供 --image 或 --dir 参数")