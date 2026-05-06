"""
创建数据集子集用于快速训练和测试
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse


def create_subset(src_dir, dst_dir, num_train=1000, num_val=200, num_test=200, seed=42):
    """
    创建数据集子集
    
    Args:
        src_dir: 源数据集目录 (yolo_format)
        dst_dir: 目标子集目录
        num_train: 训练集图像数量
        num_val: 验证集图像数量
        num_test: 测试集图像数量
        seed: 随机种子
    """
    random.seed(seed)
    
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    # 创建目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        (dst_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    subset_stats = {}
    
    for split, num_samples in [('train', num_train), ('val', num_val), ('test', num_test)]:
        src_img_dir = src_dir / 'images' / split
        src_label_dir = src_dir / 'labels' / split
        
        if not src_img_dir.exists():
            print(f"警告: {src_img_dir} 不存在，跳过")
            continue
        
        # 获取所有图像文件
        images = list(src_img_dir.glob('*.jpg')) + list(src_img_dir.glob('*.png'))
        
        if len(images) == 0:
            print(f"警告: {split} 中没有找到图像")
            continue
        
        # 随机采样
        if len(images) > num_samples:
            selected = random.sample(images, num_samples)
        else:
            selected = images
            print(f"注意: {split} 只有 {len(images)} 张图像，少于请求的 {num_samples}")
        
        # 复制文件
        copied = 0
        for img_path in tqdm(selected, desc=f"复制 {split} 数据"):
            # 复制图像
            dst_img_path = dst_dir / 'images' / split / img_path.name
            shutil.copy2(img_path, dst_img_path)
            
            # 复制对应的标注文件
            label_name = img_path.stem + '.txt'
            src_label_path = src_label_dir / label_name
            dst_label_path = dst_dir / 'labels' / split / label_name
            
            if src_label_path.exists():
                shutil.copy2(src_label_path, dst_label_path)
                copied += 1
        
        subset_stats[split] = copied
        print(f"{split}: 已复制 {copied} 对图像-标注")
    
    # 创建 YAML 配置文件
    yaml_content = f"""# Anti-UAV 红外无人机检测数据集 - 子集 ({num_train}/{num_val}/{num_test})
path: {dst_dir.absolute()}  # 数据集根目录
train: images/train  # 训练集图像路径
val: images/val  # 验证集图像路径
test: images/test  # 测试集图像路径

# 类别信息
names:
  0: UAV  # 无人机
"""
    
    yaml_path = dst_dir / 'anti_uav_subset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n子集创建完成！")
    print(f"配置文件: {yaml_path}")
    print(f"统计: {subset_stats}")
    
    return yaml_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='创建数据集子集')
    parser.add_argument('--src', type=str, default='./yolo_format', help='源数据集目录')
    parser.add_argument('--dst', type=str, default='./yolo_subset', help='目标子集目录')
    parser.add_argument('--train', type=int, default=1000, help='训练集样本数')
    parser.add_argument('--val', type=int, default=200, help='验证集样本数')
    parser.add_argument('--test', type=int, default=200, help='测试集样本数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    create_subset(args.src, args.dst, args.train, args.val, args.test, args.seed)