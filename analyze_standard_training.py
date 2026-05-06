"""
Standard模型训练日志分析脚本
单独分析Standard训练日志，生成详细的训练曲线和统计报告
"""

import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_log_file(log_path):
    """解析训练日志文件"""
    data = defaultdict(list)
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('[train:'):
                continue
            
            # 解析训练日志行
            match = re.match(
                r'\[train:\s*(\d+),\s*(\d+)\s*/\s*(\d+)\]\s*'
                r'FPS:\s*([\d.]+)\s*\([\d.]+\)\s*'
                r',\s*Loss/total:\s*([\d.nan]+)\s*'
                r',\s*Loss/giou:\s*([\d.]+)\s*'
                r',\s*Loss/l1:\s*([\d.]+)\s*'
                r',\s*Loss/location:\s*([\d.nan]+)\s*'
                r',\s*Loss/task_class:\s*([\d.]+)\s*'
                r',\s*IoU:\s*([\d.]+)\s*'
                r',\s*Precision:\s*([\d.]+)\s*'
                r',\s*Recall:\s*([\d.]+)',
                line
            )
            
            if match:
                epoch = int(match.group(1))
                iteration = int(match.group(2))
                total_iter = int(match.group(3))
                fps = float(match.group(4))
                loss_total = match.group(5)
                loss_giou = float(match.group(6))
                loss_l1 = float(match.group(7))
                loss_loc = match.group(8)
                loss_task = float(match.group(9))
                iou = float(match.group(10))
                precision = float(match.group(11))
                recall = float(match.group(12))
                
                # 处理 nan
                loss_total = float(loss_total) if loss_total != 'nan' else np.nan
                loss_loc = float(loss_loc) if loss_loc != 'nan' else np.nan
                
                # 计算全局step
                global_step = (epoch - 1) * total_iter + iteration
                
                data['epoch'].append(epoch)
                data['iteration'].append(iteration)
                data['global_step'].append(global_step)
                data['fps'].append(fps)
                data['loss_total'].append(loss_total)
                data['loss_giou'].append(loss_giou)
                data['loss_l1'].append(loss_l1)
                data['loss_location'].append(loss_loc)
                data['loss_task_class'].append(loss_task)
                data['iou'].append(iou)
                data['precision'].append(precision)
                data['recall'].append(recall)
    
    return dict(data)


def compute_epoch_stats(data):
    """计算每个epoch的统计信息"""
    epoch_stats = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(data['epoch'])):
        epoch = data['epoch'][i]
        for key in ['loss_total', 'loss_giou', 'loss_l1', 'loss_location', 
                    'loss_task_class', 'iou', 'precision', 'recall', 'fps']:
            val = data[key][i]
            if not np.isnan(val):
                epoch_stats[epoch][key].append(val)
    
    # 计算每个epoch的平均值、最小值、最大值
    result = {
        'epoch': [],
        'loss_total_mean': [], 'loss_total_min': [], 'loss_total_max': [],
        'loss_giou_mean': [], 'loss_giou_min': [], 'loss_giou_max': [],
        'loss_l1_mean': [], 'loss_l1_min': [], 'loss_l1_max': [],
        'loss_location_mean': [], 'loss_location_min': [], 'loss_location_max': [],
        'iou_mean': [], 'iou_min': [], 'iou_max': [],
        'precision_mean': [], 'precision_min': [], 'precision_max': [],
        'recall_mean': [], 'recall_min': [], 'recall_max': [],
        'fps_mean': [],
    }
    
    for epoch in sorted(epoch_stats.keys()):
        result['epoch'].append(epoch)
        
        for metric in ['loss_total', 'loss_giou', 'loss_l1', 'loss_location', 
                       'iou', 'precision', 'recall', 'fps']:
            values = epoch_stats[epoch][metric]
            if values:
                result[f'{metric}_mean'].append(np.mean(values))
                if metric != 'fps':  # FPS不需要min/max
                    result[f'{metric}_min'].append(np.min(values))
                    result[f'{metric}_max'].append(np.max(values))
            else:
                result[f'{metric}_mean'].append(np.nan)
                if metric != 'fps':
                    result[f'{metric}_min'].append(np.nan)
                    result[f'{metric}_max'].append(np.nan)
    
    return result


def plot_training_curves(stats, output_dir='standard_training_analysis'):
    """绘制训练曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    epochs = stats['epoch']
    
    # 1. 损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Standard Model - Training Loss Curves', fontsize=16)
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, stats['loss_total_mean'], 'b-', linewidth=2, label='Mean')
    ax.fill_between(epochs, stats['loss_total_min'], stats['loss_total_max'], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # GIoU Loss
    ax = axes[0, 1]
    ax.plot(epochs, stats['loss_giou_mean'], 'r-', linewidth=2, label='Mean')
    ax.fill_between(epochs, stats['loss_giou_min'], stats['loss_giou_max'], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('GIoU Loss')
    ax.set_title('GIoU Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # L1 Loss
    ax = axes[1, 0]
    ax.plot(epochs, stats['loss_l1_mean'], 'g-', linewidth=2, label='Mean')
    ax.fill_between(epochs, stats['loss_l1_min'], stats['loss_l1_max'], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('L1 Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Location Loss
    ax = axes[1, 1]
    ax.plot(epochs, stats['loss_location_mean'], 'm-', linewidth=2, label='Mean')
    ax.fill_between(epochs, stats['loss_location_min'], stats['loss_location_max'], alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Location Loss')
    ax.set_title('Location Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/loss_curves.png")
    plt.close()
    
    # 2. 性能指标曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Standard Model - Training Performance Metrics', fontsize=16)
    
    # IoU
    ax = axes[0, 0]
    ax.plot(epochs, stats['iou_mean'], 'b-', linewidth=2, label='Mean IoU')
    ax.fill_between(epochs, stats['iou_min'], stats['iou_max'], alpha=0.3, label='Min-Max Range')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='IoU=0.5')
    ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='IoU=0.8')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('IoU (Intersection over Union)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Precision
    ax = axes[0, 1]
    ax.plot(epochs, stats['precision_mean'], 'g-', linewidth=2, label='Mean Precision')
    ax.fill_between(epochs, stats['precision_min'], stats['precision_max'], alpha=0.3)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Precision=0.9')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # Recall
    ax = axes[1, 0]
    ax.plot(epochs, stats['recall_mean'], 'r-', linewidth=2, label='Mean Recall')
    ax.fill_between(epochs, stats['recall_min'], stats['recall_max'], alpha=0.3)
    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Recall=0.9')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1])
    
    # FPS
    ax = axes[1, 1]
    ax.plot(epochs, stats['fps_mean'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FPS')
    ax.set_title('Training Speed (FPS)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_curves.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/performance_curves.png")
    plt.close()
    
    # 3. 综合对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(epochs, stats['iou_mean'], 'b-', linewidth=2, label='IoU', marker='o', markersize=3)
    ax.plot(epochs, stats['precision_mean'], 'g-', linewidth=2, label='Precision', marker='s', markersize=3)
    ax.plot(epochs, stats['recall_mean'], 'r-', linewidth=2, label='Recall', marker='^', markersize=3)
    
    # 标记学习率衰减点
    lr_drop_epoch = 90
    ax.axvline(x=lr_drop_epoch, color='orange', linestyle='--', alpha=0.7, label=f'LR Drop (Epoch {lr_drop_epoch})')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Standard Model - Training Metrics Over Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/combined_metrics.png")
    plt.close()


def print_analysis_report(stats):
    """打印分析报告"""
    print("\n" + "="*80)
    print("Standard Model Training Analysis Report")
    print("="*80)
    
    epochs = stats['epoch']
    
    # 1. 基本信息
    print("\n[Basic Information]")
    print(f"Total Epochs: {max(epochs)}")
    print(f"Total Iterations: {len(epochs)}")
    
    # 2. 最终性能
    print("\n[Final Performance (Last Epoch)]")
    print("-"*80)
    print(f"{'Metric':<20} {'Mean':<15} {'Min':<15} {'Max':<15}")
    print("-"*80)
    for metric in ['loss_total', 'loss_giou', 'loss_l1', 'loss_location', 
                   'iou', 'precision', 'recall']:
        mean_val = stats[f'{metric}_mean'][-1]
        min_val = stats[f'{metric}_min'][-1] if stats[f'{metric}_min'] else np.nan
        max_val = stats[f'{metric}_max'][-1] if stats[f'{metric}_max'] else np.nan
        print(f"{metric:<20} {mean_val:<15.4f} {min_val:<15.4f} {max_val:<15.4f}")
    print(f"{'fps':<20} {stats['fps_mean'][-1]:<15.2f}")
    print("-"*80)
    
    # 3. 收敛分析
    print("\n[Convergence Analysis]")
    
    # IoU达到0.5的epoch
    iou_05_idx = next((i for i, x in enumerate(stats['iou_mean']) if x > 0.5), None)
    if iou_05_idx:
        print(f"IoU > 0.5 reached at epoch {epochs[iou_05_idx]}")
    
    # IoU达到0.8的epoch
    iou_08_idx = next((i for i, x in enumerate(stats['iou_mean']) if x > 0.8), None)
    if iou_08_idx:
        print(f"IoU > 0.8 reached at epoch {epochs[iou_08_idx]}")
    
    # Precision达到0.9的epoch
    prec_09_idx = next((i for i, x in enumerate(stats['precision_mean']) if x > 0.9), None)
    if prec_09_idx:
        print(f"Precision > 0.9 reached at epoch {epochs[prec_09_idx]}")
    
    # 4. 学习率衰减影响
    lr_drop_epoch = 90
    if lr_drop_epoch in epochs:
        idx_before = epochs.index(lr_drop_epoch) - 1
        idx_after = epochs.index(lr_drop_epoch)
        
        print(f"\n[Learning Rate Drop Impact (Epoch {lr_drop_epoch})]")
        print(f"IoU: {stats['iou_mean'][idx_before]:.4f} -> {stats['iou_mean'][idx_after]:.4f} "
              f"({stats['iou_mean'][idx_after] - stats['iou_mean'][idx_before]:+.4f})")
        print(f"Precision: {stats['precision_mean'][idx_before]:.4f} -> {stats['precision_mean'][idx_after]:.4f} "
              f"({stats['precision_mean'][idx_after] - stats['precision_mean'][idx_before]:+.4f})")
    
    # 5. 训练稳定性
    print("\n[Training Stability]")
    iou_std = np.std(stats['iou_mean'])
    prec_std = np.std(stats['precision_mean'])
    print(f"IoU std: {iou_std:.4f}")
    print(f"Precision std: {prec_std:.4f}")
    
    # 6. 关键阶段
    print("\n[Key Training Stages]")
    stages = [1, 10, 50, 90, 120, 150, 180]
    print(f"{'Epoch':<10} {'IoU':<10} {'Precision':<12} {'Recall':<10} {'Loss':<10}")
    print("-"*60)
    for stage in stages:
        if stage in epochs:
            idx = epochs.index(stage)
            print(f"{stage:<10} {stats['iou_mean'][idx]:<10.4f} "
                  f"{stats['precision_mean'][idx]:<12.4f} "
                  f"{stats['recall_mean'][idx]:<10.4f} "
                  f"{stats['loss_total_mean'][idx]:<10.4f}")
    
    print("\n" + "="*80)


def filter_first_n_epochs(stats, n=100):
    """只保留前n个epoch的数据"""
    filtered = {}
    for key, values in stats.items():
        if key == 'epoch':
            filtered[key] = [v for v in values if v <= n]
        else:
            # 根据epoch长度截取对应的数据
            epoch_count = len([e for e in stats['epoch'] if e <= n])
            filtered[key] = values[:epoch_count]
    return filtered


def main():
    """主函数"""
    log_path = r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_standard\logs\sutrack-sutrack_b224_antiuav_standard.log'
    output_dir = 'standard_training_analysis'
    MAX_EPOCHS = 100  # 只分析前100个epoch
    
    print("="*80)
    print(f"Standard Model Training Log Analysis (First {MAX_EPOCHS} Epochs)")
    print("="*80)
    
    if not os.path.exists(log_path):
        print(f"[Error] Log file not found: {log_path}")
        return
    
    print(f"\nParsing log file: {log_path}")
    data = parse_log_file(log_path)
    
    if not data['epoch']:
        print("[Error] No training data found in log file!")
        return
    
    print(f"Parsed {len(data['epoch'])} training records")
    print(f"Epochs: {min(data['epoch'])} - {max(data['epoch'])}")
    
    print("\nComputing epoch statistics...")
    stats = compute_epoch_stats(data)
    
    # 只保留前100个epoch
    print(f"\nFiltering to first {MAX_EPOCHS} epochs...")
    stats = filter_first_n_epochs(stats, MAX_EPOCHS)
    
    print(f"\nGenerating plots (Epochs 1-{MAX_EPOCHS})...")
    plot_training_curves(stats, output_dir)
    
    print_analysis_report(stats)
    
    # 保存数据
    with open(os.path.join(output_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n[OK] Statistics saved to: {output_dir}/training_stats.json")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
