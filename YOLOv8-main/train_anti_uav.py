"""
Anti-UAV 红外无人机检测 - YOLOv8 训练脚本
支持分阶段训练策略和红外图像特定增强
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from datetime import datetime

# 添加ultralytics到路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from ultralytics.yolo.utils import LOGGER
import cv2
import numpy as np
import random


def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IRAugmentation:
    """红外图像特定数据增强"""
    
    @staticmethod
    def apply_ir_brightness(img, factor=0.3):
        """红外亮度调整"""
        factor = random.uniform(1 - factor, 1 + factor)
        return np.clip(img * factor, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_ir_contrast(img, factor=0.3):
        """红外对比度调整"""
        factor = random.uniform(1 - factor, 1 + factor)
        mean = img.mean()
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_ir_noise(img, intensity=0.05):
        """添加高斯噪声（模拟红外传感器噪声）"""
        noise = np.random.normal(0, intensity * 255, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_ir_blur(img, prob=0.1):
        """随机模糊（模拟失焦）"""
        if random.random() < prob:
            kernel_size = random.choice([3, 5])
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img
    
    @staticmethod
    def apply_thermal_crossover_simulation(img, prob=0.2):
        """模拟热交叉效应（红外特有挑战）"""
        if random.random() < prob:
            # 随机反转部分区域的亮度（模拟目标与背景热交叉）
            h, w = img.shape[:2]
            x, y = random.randint(0, w//2), random.randint(0, h//2)
            bw, bh = random.randint(w//4, w//2), random.randint(h//4, h//2)
            
            region = img[y:y+bh, x:x+bw].copy()
            region = 255 - region  # 亮度反转
            img[y:y+bh, x:x+bw] = region
        return img


def train_phase1(model, data_yaml, epochs=50, batch=16, imgsz=640, device='0'):
    """
    第一阶段训练：冻结骨干网络，训练检测头
    适用于在预训练模型基础上快速适应新数据集
    """
    LOGGER.info("=" * 60)
    LOGGER.info("第一阶段训练：冻结骨干网络，训练检测头")
    LOGGER.info("=" * 60)
    
    # 手动冻结骨干网络层
    LOGGER.info("冻结前10层（骨干网络）...")
    for i, (name, param) in enumerate(model.model.named_parameters()):
        if i < 10:  # 冻结前10层
            param.requires_grad = False
            LOGGER.info(f"  冻结: {name}")
    
    # 训练参数
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        
        # 学习率设置（较低的学习率）
        lr0=0.001,
        lrf=0.01,
        
        # 优化器
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.4,
        degrees=3.0,
        translate=0.1,
        scale=0.3,
        shear=1.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        
        # 其他设置
        cos_lr=True,
        patience=20,
        save=True,
        project='runs/detect',
        name='anti_uav_phase1',
        exist_ok=False,
        pretrained=True,
        seed=42,
        deterministic=True,
        verbose=True
    )
    
    # 解冻所有层
    LOGGER.info("解冻所有层...")
    for param in model.model.parameters():
        param.requires_grad = True
    
    return results, model


def train_phase2(model, data_yaml, epochs=150, batch=16, imgsz=640, device='0'):
    """
    第二阶段训练：解冻全部网络，端到端微调
    在全数据集上进行精细调整
    """
    LOGGER.info("=" * 60)
    LOGGER.info("第二阶段训练：全网络端到端微调")
    LOGGER.info("=" * 60)
    
    # 确保所有层都解冻
    for param in model.model.parameters():
        param.requires_grad = True
    
    # 训练参数
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        
        # 学习率设置（正常学习率）
        lr0=0.01,
        lrf=0.01,
        
        # 优化器
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.6,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # 其他设置
        cos_lr=True,
        patience=30,
        save=True,
        project='runs/detect',
        name='anti_uav_phase2',
        exist_ok=False,
        seed=42,
        deterministic=True,
        verbose=True
    )
    
    return results, model


def train_single_stage(model_path, data_yaml, epochs=200, batch=16, imgsz=640, device='0'):
    """
    单阶段训练（不使用分阶段策略）
    适用于从头训练或快速实验
    """
    LOGGER.info("=" * 60)
    LOGGER.info("单阶段训练模式")
    LOGGER.info("=" * 60)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        
        # 学习率
        lr0=0.01,
        lrf=0.01,
        
        # 优化器
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        
        # 数据增强（简化版，避免兼容性问题）
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,  # 禁用mixup避免兼容问题
        
        # 其他设置
        cos_lr=True,
        patience=30,
        save=True,
        project='runs/detect',
        name='anti_uav_single_stage',
        exist_ok=False,
        pretrained=True,
        seed=42,
        deterministic=True,
        verbose=True
    )
    
    return results, model


def validate_model(model, data_yaml, split='val'):
    """验证模型性能"""
    LOGGER.info(f"\n{'=' * 60}")
    LOGGER.info(f"验证模型 - {split}集")
    LOGGER.info(f"{'=' * 60}\n")
    
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=640,
        batch=16,
        conf=0.001,  # 低置信度阈值以获得完整的PR曲线
        iou=0.6,     # NMS IoU阈值
        max_det=300,
        save_json=True,
        save_hybrid=True,
        plots=True
    )
    
    return results


def export_model(model, format='onnx'):
    """导出模型"""
    LOGGER.info(f"\n{'=' * 60}")
    LOGGER.info(f"导出模型为 {format} 格式")
    LOGGER.info(f"{'=' * 60}\n")
    
    model.export(
        format=format,
        imgsz=640,
        half=True,      # FP16半精度
        int8=False,     # 不量化
        simplify=True,  # 简化ONNX
        opset=13        # ONNX opset版本
    )


def main():
    parser = argparse.ArgumentParser(description='Anti-UAV YOLOv8 Training')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'phased', 'resume'],
                       help='训练模式: single(单阶段), phased(分阶段), resume(恢复训练)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       help='预训练模型路径')
    parser.add_argument('--data', type=str, default='data/AntI-UAV/yolo_format/anti_uav.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='0',
                       help='GPU设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--validate', action='store_true',
                       help='训练后验证')
    parser.add_argument('--export', action='store_true',
                       help='训练后导出模型')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 记录训练开始时间
    start_time = datetime.now()
    LOGGER.info(f"训练开始时间: {start_time}")
    LOGGER.info(f"训练模式: {args.mode}")
    LOGGER.info(f"模型: {args.model}")
    LOGGER.info(f"数据集: {args.data}")
    
    # 检查数据集配置是否存在
    if not Path(args.data).exists():
        LOGGER.error(f"数据集配置文件不存在: {args.data}")
        LOGGER.info("请先运行数据预处理脚本: python data/AntI-UAV/prepare_dataset.py")
        return
    
    # 根据模式执行训练
    if args.mode == 'single':
        # 单阶段训练
        results, model = train_single_stage(
            args.model, args.data, 
            epochs=args.epochs, 
            batch=args.batch, 
            imgsz=args.imgsz, 
            device=args.device
        )
        
    elif args.mode == 'phased':
        # 分阶段训练
        # 阶段1：冻结骨干网络
        model = YOLO(args.model)
        results1, model = train_phase1(
            model, args.data,
            epochs=50,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
        
        # 阶段2：全网络微调
        # 加载阶段1的最佳权重
        best_weights = 'runs/detect/anti_uav_phase1/weights/best.pt'
        if Path(best_weights).exists():
            model = YOLO(best_weights)
        
        results2, model = train_phase2(
            model, args.data,
            epochs=args.epochs - 50,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
        
    elif args.mode == 'resume':
        # 恢复训练
        LOGGER.info(f"恢复训练，目标轮次: {args.epochs}")
        model = YOLO(args.model)
        
        # 如果指定了新的轮次，使用新轮次
        if args.epochs != 200:  # 默认200，如果修改了则使用新值
            results = model.train(
                resume=True,
                epochs=args.epochs
            )
        else:
            results = model.train(resume=True)
    
    # 训练结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    LOGGER.info(f"\n训练结束时间: {end_time}")
    LOGGER.info(f"总训练时长: {duration}")
    
    # 验证
    if args.validate and 'model' in locals():
        validate_model(model, args.data, split='val')
        validate_model(model, args.data, split='test')
    
    # 导出模型
    if args.export and 'model' in locals():
        export_model(model, format='onnx')
        export_model(model, format='engine')  # TensorRT
    
    LOGGER.info("\n训练流程完成！")


if __name__ == '__main__':
    main()
