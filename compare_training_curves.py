"""
训练日志对比分析脚本
对比 Standard 和 CLAHE 两种配置的训练曲线
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
            # 格式: [train: epoch, iter / total] FPS: xx  ,  Loss/total: xx  ,  ...
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


def compute_epoch_average(data):
    """计算每个epoch的平均值"""
    epoch_data = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(data['epoch'])):
        epoch = data['epoch'][i]
        for key in ['loss_total', 'loss_giou', 'loss_l1', 'loss_location', 
                    'loss_task_class', 'iou', 'precision', 'recall', 'fps']:
            if not np.isnan(data[key][i]):
                epoch_data[epoch][key].append(data[key][i])
    
    # 计算平均值
    result = defaultdict(list)
    for epoch in sorted(epoch_data.keys()):
        result['epoch'].append(epoch)
        for key in ['loss_total', 'loss_giou', 'loss_l1', 'loss_location',
                    'loss_task_class', 'iou', 'precision', 'recall', 'fps']:
            values = epoch_data[epoch][key]
            result[key].append(np.mean(values) if values else np.nan)
    
    return dict(result)


def plot_comparison(standard_data, clahe_data, output_dir='training_comparison'):
    """绘制对比曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算epoch平均值
    std_epoch = compute_epoch_average(standard_data)
    clahe_epoch = compute_epoch_average(clahe_data)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 总损失对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Curves Comparison: Standard vs CLAHE', fontsize=16)
    
    # Loss Total
    ax = axes[0, 0]
    ax.plot(std_epoch['epoch'], std_epoch['loss_total'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['loss_total'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss GIoU
    ax = axes[0, 1]
    ax.plot(std_epoch['epoch'], std_epoch['loss_giou'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['loss_giou'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('GIoU Loss')
    ax.set_title('GIoU Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss L1
    ax = axes[0, 2]
    ax.plot(std_epoch['epoch'], std_epoch['loss_l1'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['loss_l1'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('L1 Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU
    ax = axes[1, 0]
    ax.plot(std_epoch['epoch'], std_epoch['iou'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['iou'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('IoU (Intersection over Union)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[1, 1]
    ax.plot(std_epoch['epoch'], std_epoch['precision'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['precision'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[1, 2]
    ax.plot(std_epoch['epoch'], std_epoch['recall'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['recall'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_comparison.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/training_curves_comparison.png")
    plt.close()
    
    # 2. 详细损失分解
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Loss Components Comparison: Standard vs CLAHE', fontsize=16)
    
    # Location Loss
    ax = axes[0, 0]
    ax.plot(std_epoch['epoch'], std_epoch['loss_location'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['loss_location'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Location Loss')
    ax.set_title('Location Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Task Class Loss
    ax = axes[0, 1]
    ax.plot(std_epoch['epoch'], std_epoch['loss_task_class'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['loss_task_class'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Task Class Loss')
    ax.set_title('Task Class Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FPS
    ax = axes[1, 0]
    ax.plot(std_epoch['epoch'], std_epoch['fps'], 'b-', label='Standard', linewidth=2)
    ax.plot(clahe_epoch['epoch'], clahe_epoch['fps'], 'r-', label='CLAHE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FPS')
    ax.set_title('Training Speed (FPS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 综合指标对比（最终epoch）
    ax = axes[1, 1]
    metrics = ['IoU', 'Precision', 'Recall']
    std_final = [std_epoch['iou'][-1], std_epoch['precision'][-1], std_epoch['recall'][-1]]
    clahe_final = [clahe_epoch['iou'][-1], clahe_epoch['precision'][-1], clahe_epoch['recall'][-1]]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, std_final, width, label='Standard', color='blue', alpha=0.7)
    ax.bar(x + width/2, clahe_final, width, label='CLAHE', color='red', alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Final Epoch Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(std_final, clahe_final)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_components_comparison.png'), dpi=300)
    print(f"[OK] Saved: {output_dir}/loss_components_comparison.png")
    plt.close()
    
    return std_epoch, clahe_epoch


def print_statistics(std_data, clahe_data, std_epoch, clahe_epoch):
    """打印统计信息"""
    print("\n" + "="*80)
    print("Training Log Comparison: Standard vs CLAHE")
    print("="*80)
    
    print("\n[Data Summary]")
    print(f"Standard - Total iterations: {len(std_data['global_step'])}, Epochs: {max(std_data['epoch'])}")
    print(f"CLAHE    - Total iterations: {len(clahe_data['global_step'])}, Epochs: {max(clahe_data['epoch'])}")
    
    print("\n[Final Epoch Comparison]")
    print("-"*80)
    print(f"{'Metric':<20} {'Standard':<15} {'CLAHE':<15} {'Diff':<15}")
    print("-"*80)
    
    metrics = [
        ('Total Loss', 'loss_total'),
        ('GIoU Loss', 'loss_giou'),
        ('L1 Loss', 'loss_l1'),
        ('IoU', 'iou'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('FPS', 'fps')
    ]
    
    for name, key in metrics:
        std_val = std_epoch[key][-1] if std_epoch[key] else 0
        clahe_val = clahe_epoch[key][-1] if clahe_epoch[key] else 0
        diff = clahe_val - std_val
        diff_pct = (diff / std_val * 100) if std_val != 0 else 0
        print(f"{name:<20} {std_val:<15.4f} {clahe_val:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%)")
    
    print("-"*80)
    
    # 收敛速度分析
    print("\n[Convergence Analysis]")
    std_iou_50 = next((i for i, x in enumerate(std_epoch['iou']) if x > 0.5), None)
    clahe_iou_50 = next((i for i, x in enumerate(clahe_epoch['iou']) if x > 0.5), None)
    
    if std_iou_50:
        print(f"Standard reached IoU>0.5 at epoch {std_epoch['epoch'][std_iou_50]}")
    if clahe_iou_50:
        print(f"CLAHE reached IoU>0.5 at epoch {clahe_epoch['epoch'][clahe_iou_50]}")
    
    # 训练稳定性分析
    print("\n[Training Stability]")
    std_loss_std = np.nanstd(std_epoch['loss_total'])
    clahe_loss_std = np.nanstd(clahe_epoch['loss_total'])
    print(f"Standard loss std: {std_loss_std:.4f}")
    print(f"CLAHE loss std: {clahe_loss_std:.4f}")
    print(f"More stable: {'Standard' if std_loss_std < clahe_loss_std else 'CLAHE'}")


def main():
    """主函数"""
    # 日志文件路径
    standard_log = r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_standard\logs\sutrack-sutrack_b224_antiuav_standard.log'
    clahe_log = r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_clahe\logs\sutrack-sutrack_b224_antiuav_clahe.log'
    
    print("Parsing Standard training log...")
    standard_data = parse_log_file(standard_log)
    
    print("Parsing CLAHE training log...")
    clahe_data = parse_log_file(clahe_log)
    
    if not standard_data['epoch']:
        print("[Error] No data parsed from Standard log!")
        return
    if not clahe_data['epoch']:
        print("[Error] No data parsed from CLAHE log!")
        return
    
    print("Generating comparison plots...")
    std_epoch, clahe_epoch = plot_comparison(standard_data, clahe_data)
    
    print_statistics(standard_data, clahe_data, std_epoch, clahe_epoch)
    
    # 保存数据
    output_dir = 'training_comparison'
    with open(os.path.join(output_dir, 'standard_epoch_data.json'), 'w') as f:
        json.dump(std_epoch, f, indent=2)
    with open(os.path.join(output_dir, 'clahe_epoch_data.json'), 'w') as f:
        json.dump(clahe_epoch, f, indent=2)
    print(f"\n[OK] Data saved to {output_dir}/")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
