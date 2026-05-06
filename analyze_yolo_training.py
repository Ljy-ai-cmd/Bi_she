"""
YOLOv8训练结果分析脚本
解析YOLOv8训练生成的results.csv，生成训练曲线和统计报告
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_csv_file(csv_path):
    """解析YOLO训练结果CSV文件"""
    data = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 解析每一行数据
            data['epoch'].append(int(row['                  epoch'].strip()))
            data['train_box_loss'].append(float(row['         train/box_loss'].strip()))
            data['train_cls_loss'].append(float(row['         train/cls_loss'].strip()))
            data['train_dfl_loss'].append(float(row['         train/dfl_loss'].strip()))
            
            # 处理可能为空的metrics
            precision = row['   metrics/precision(B)'].strip()
            data['precision'].append(float(precision) if precision and precision != '0' else 0.0)
            
            recall = row['      metrics/recall(B)'].strip()
            data['recall'].append(float(recall) if recall else 0.0)
            
            map50 = row['       metrics/mAP50(B)'].strip()
            data['map50'].append(float(map50) if map50 else 0.0)
            
            map50_95 = row['    metrics/mAP50-95(B)'].strip()
            data['map50_95'].append(float(map50_95) if map50_95 else 0.0)
            
            # 学习率
            data['lr_pg0'].append(float(row['                 lr/pg0'].strip()))
    
    return dict(data)


def plot_training_curves(data, output_dir='yolo_training_analysis'):
    """绘制YOLO训练曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = data['epoch']
    
    # 1. 训练损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLOv8 Training Loss Curves', fontsize=16)
    
    # Box Loss
    ax = axes[0, 0]
    ax.plot(epochs, data['train_box_loss'], 'b-', linewidth=2, label='Box Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Box Loss')
    ax.set_title('Training Box Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Cls Loss
    ax = axes[0, 1]
    ax.plot(epochs, data['train_cls_loss'], 'r-', linewidth=2, label='Cls Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cls Loss')
    ax.set_title('Training Classification Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # DFL Loss
    ax = axes[1, 0]
    ax.plot(epochs, data['train_dfl_loss'], 'g-', linewidth=2, label='DFL Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('DFL Loss')
    ax.set_title('Training DFL Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Total Loss (approximate)
    ax = axes[1, 1]
    total_loss = np.array(data['train_box_loss']) + np.array(data['train_cls_loss']) + np.array(data['train_dfl_loss'])
    ax.plot(epochs, total_loss, 'purple', linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Total Loss (Sum)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yolo_loss_curves.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/yolo_loss_curves.png")
    plt.close()
    
    # 2. 性能指标曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLOv8 Training Performance Metrics', fontsize=16)
    
    # Precision
    ax = axes[0, 0]
    valid_prec = [(e, p) for e, p in zip(epochs, data['precision']) if p > 0]
    if valid_prec:
        valid_epochs, valid_vals = zip(*valid_prec)
        ax.plot(valid_epochs, valid_vals, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision (BBox)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Recall
    ax = axes[0, 1]
    valid_rec = [(e, r) for e, r in zip(epochs, data['recall']) if r > 0]
    if valid_rec:
        valid_epochs, valid_vals = zip(*valid_rec)
        ax.plot(valid_epochs, valid_vals, 'g-', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall (BBox)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # mAP50
    ax = axes[1, 0]
    valid_map50 = [(e, m) for e, m in zip(epochs, data['map50']) if m > 0]
    if valid_map50:
        valid_epochs, valid_vals = zip(*valid_map50)
        ax.plot(valid_epochs, valid_vals, 'r-', linewidth=2, marker='^', markersize=3)
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='mAP50=0.9')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50')
    ax.set_title('mAP@0.5')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.05])
    
    # mAP50-95
    ax = axes[1, 1]
    valid_map50_95 = [(e, m) for e, m in zip(epochs, data['map50_95']) if m > 0]
    if valid_map50_95:
        valid_epochs, valid_vals = zip(*valid_map50_95)
        ax.plot(valid_epochs, valid_vals, 'purple', linewidth=2, marker='d', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.set_title('mAP@0.5:0.95')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.8])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yolo_performance_curves.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/yolo_performance_curves.png")
    plt.close()
    
    # 3. 学习率曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, data['lr_pg0'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('YOLOv8 Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yolo_lr_schedule.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/yolo_lr_schedule.png")
    plt.close()
    
    # 4. 综合对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_prec = [(e, p) for e, p in zip(epochs, data['precision']) if p > 0]
    valid_rec = [(e, r) for e, r in zip(epochs, data['recall']) if r > 0]
    valid_map50 = [(e, m) for e, m in zip(epochs, data['map50']) if m > 0]
    
    if valid_prec:
        valid_epochs, valid_vals = zip(*valid_prec)
        ax.plot(valid_epochs, valid_vals, 'b-', linewidth=2, label='Precision', marker='o', markersize=2)
    if valid_rec:
        valid_epochs, valid_vals = zip(*valid_rec)
        ax.plot(valid_epochs, valid_vals, 'g-', linewidth=2, label='Recall', marker='s', markersize=2)
    if valid_map50:
        valid_epochs, valid_vals = zip(*valid_map50)
        ax.plot(valid_epochs, valid_vals, 'r-', linewidth=2, label='mAP50', marker='^', markersize=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('YOLOv8 Training Metrics Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'yolo_combined_metrics.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/yolo_combined_metrics.png")
    plt.close()


def print_analysis_report(data):
    """打印YOLO训练分析报告"""
    print("\n" + "="*80)
    print("YOLOv8 Training Analysis Report")
    print("="*80)
    
    epochs = data['epoch']
    
    # 1. 基本信息
    print("\n[Basic Information]")
    print(f"Total Epochs: {max(epochs)}")
    print(f"Total Iterations: {len(epochs)}")
    
    # 2. 最终性能
    print("\n[Final Performance (Last Epoch)]")
    print("-"*60)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-"*60)
    print(f"{'Box Loss':<20} {data['train_box_loss'][-1]:<15.4f}")
    print(f"{'Cls Loss':<20} {data['train_cls_loss'][-1]:<15.4f}")
    print(f"{'DFL Loss':<20} {data['train_dfl_loss'][-1]:<15.4f}")
    total_loss = data['train_box_loss'][-1] + data['train_cls_loss'][-1] + data['train_dfl_loss'][-1]
    print(f"{'Total Loss':<20} {total_loss:<15.4f}")
    print(f"{'Precision':<20} {data['precision'][-1]:<15.4f}")
    print(f"{'Recall':<20} {data['recall'][-1]:<15.4f}")
    print(f"{'mAP50':<20} {data['map50'][-1]:<15.4f}")
    print(f"{'mAP50-95':<20} {data['map50_95'][-1]:<15.4f}")
    print(f"{'Learning Rate':<20} {data['lr_pg0'][-1]:<15.6f}")
    print("-"*60)
    
    # 3. 收敛分析
    print("\n[Convergence Analysis]")
    
    # 找到mAP50首次超过0.9的epoch
    map50_09_idx = next((i for i, x in enumerate(data['map50']) if x > 0.9), None)
    if map50_09_idx:
        print(f"mAP50 > 0.9 reached at epoch {epochs[map50_09_idx]}")
    
    # 找到precision首次超过0.9的epoch
    prec_09_idx = next((i for i, x in enumerate(data['precision']) if x > 0.9), None)
    if prec_09_idx:
        print(f"Precision > 0.9 reached at epoch {epochs[prec_09_idx]}")
    
    # 找到recall首次超过0.9的epoch
    rec_09_idx = next((i for i, x in enumerate(data['recall']) if x > 0.9), None)
    if rec_09_idx:
        print(f"Recall > 0.9 reached at epoch {epochs[rec_09_idx]}")
    
    # 4. 最佳性能
    print("\n[Best Performance]")
    best_map50_idx = np.argmax(data['map50'])
    print(f"Best mAP50: {data['map50'][best_map50_idx]:.4f} at epoch {epochs[best_map50_idx]}")
    
    best_prec_idx = np.argmax(data['precision'])
    print(f"Best Precision: {data['precision'][best_prec_idx]:.4f} at epoch {epochs[best_prec_idx]}")
    
    best_rec_idx = np.argmax(data['recall'])
    print(f"Best Recall: {data['recall'][best_rec_idx]:.4f} at epoch {epochs[best_rec_idx]}")
    
    # 5. 关键阶段
    print("\n[Key Training Stages]")
    stages = [1, 10, 25, 50, 75, 100]
    if max(epochs) > 100:
        stages.extend([max(epochs)])
    
    print(f"{'Epoch':<10} {'Box Loss':<12} {'Cls Loss':<12} {'mAP50':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*70)
    for stage in stages:
        if stage <= max(epochs):
            idx = epochs.index(stage) if stage in epochs else -1
            if idx >= 0:
                print(f"{stage:<10} {data['train_box_loss'][idx]:<12.4f} "
                      f"{data['train_cls_loss'][idx]:<12.4f} "
                      f"{data['map50'][idx]:<10.4f} "
                      f"{data['precision'][idx]:<12.4f} "
                      f"{data['recall'][idx]:<10.4f}")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    csv_path = r'E:\biyesheji\SUTrack-main11\YOLOv8-main\runs\detect\anti_uav_single_stage16\results.csv'
    output_dir = 'yolo_training_analysis'
    
    print("="*80)
    print("YOLOv8 Training Results Analysis")
    print("="*80)
    
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: {csv_path}")
        return
    
    print(f"\nParsing CSV file: {csv_path}")
    data = parse_csv_file(csv_path)
    
    if not data['epoch']:
        print("[Error] No training data found in CSV file!")
        return
    
    print(f"Parsed {len(data['epoch'])} epochs")
    print(f"Epochs: {min(data['epoch'])} - {max(data['epoch'])}")
    
    print("\nGenerating plots...")
    plot_training_curves(data, output_dir)
    
    print_analysis_report(data)
    
    # 保存数据
    with open(os.path.join(output_dir, 'yolo_training_stats.json'), 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n[OK] Statistics saved to: {output_dir}/yolo_training_stats.json")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
