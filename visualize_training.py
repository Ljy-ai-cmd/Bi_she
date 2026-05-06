#!/usr/bin/env python3
"""
SUTrack 训练结果可视化分析
生成训练曲线图表
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_log_file(log_path):
    """解析训练日志文件"""
    if not os.path.exists(log_path):
        return None
    
    results = {}
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # 正则表达式模式
    patterns = {
        'epoch': re.compile(r'\[train:\s*(\d+),\s*(\d+)\s*/\s*(\d+)\]'),
        'loss': re.compile(r'Loss/total:\s*([\d.]+)'),
        'giou': re.compile(r'Loss/giou:\s*([\d.]+)'),
        'l1': re.compile(r'Loss/l1:\s*([\d.]+)'),
        'location': re.compile(r'Loss/location:\s*([\d.]+)'),
        'iou': re.compile(r'IoU:\s*([\d.]+)'),
        'precision': re.compile(r'Precision:\s*([\d.]+)'),
        'recall': re.compile(r'Recall:\s*([\d.]+)'),
        'fps': re.compile(r'FPS:\s*([\d.]+)'),
    }
    
    for line in lines:
        epoch_match = patterns['epoch'].search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            step = int(epoch_match.group(2))
            total_steps = int(epoch_match.group(3))
            
            if epoch not in results:
                results[epoch] = {
                    'steps': [], 'loss': [], 'giou': [], 'l1': [], 'location': [],
                    'iou': [], 'precision': [], 'recall': [], 'fps': []
                }
            
            data = {'step': step, 'total_steps': total_steps}
            for key, pattern in patterns.items():
                if key != 'epoch':
                    match = pattern.search(line)
                    data[key] = float(match.group(1)) if match else None
            
            results[epoch]['steps'].append(step)
            for key in ['loss', 'giou', 'l1', 'location', 'iou', 'precision', 'recall', 'fps']:
                if data.get(key) is not None:
                    results[epoch][key].append(data[key])
    
    # 计算每个epoch的平均值
    epoch_summary = {}
    for epoch, data in results.items():
        epoch_summary[epoch] = {}
        for key in ['loss', 'giou', 'l1', 'location', 'iou', 'precision', 'recall', 'fps']:
            if data[key]:
                epoch_summary[epoch][key] = np.mean(data[key])
    
    return epoch_summary


def plot_training_curves(exp_data, output_dir='visualization'):
    """绘制训练曲线"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 准备数据
    experiments = list(exp_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Loss 曲线
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SUTrack Training Curves', fontsize=16, fontweight='bold')
    
    # Total Loss
    ax = axes[0, 0]
    for i, (exp_name, data) in enumerate(exp_data.items()):
        epochs = sorted(data.keys())
        losses = [data[e].get('loss', 0) for e in epochs]
        ax.plot(epochs, losses, label=exp_name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IoU
    ax = axes[0, 1]
    for i, (exp_name, data) in enumerate(exp_data.items()):
        epochs = sorted(data.keys())
        ious = [data[e].get('iou', 0) for e in epochs]
        ax.plot(epochs, ious, label=exp_name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('IoU Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision & Recall
    ax = axes[1, 0]
    for i, (exp_name, data) in enumerate(exp_data.items()):
        epochs = sorted(data.keys())
        precisions = [data[e].get('precision', 0) for e in epochs]
        recalls = [data[e].get('recall', 0) for e in epochs]
        ax.plot(epochs, precisions, label=f'{exp_name} (Precision)', 
                color=colors[i % len(colors)], linewidth=2, linestyle='-')
        ax.plot(epochs, recalls, label=f'{exp_name} (Recall)', 
                color=colors[i % len(colors)], linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FPS
    ax = axes[1, 1]
    for i, (exp_name, data) in enumerate(exp_data.items()):
        epochs = sorted(data.keys())
        fps_values = [data[e].get('fps', 0) for e in epochs]
        ax.plot(epochs, fps_values, label=exp_name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FPS')
    ax.set_title('Training Speed (FPS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/training_curves.png")
    plt.close()
    
    # 2. 单独的损失分解图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Loss Components', fontsize=14, fontweight='bold')
    
    for i, (exp_name, data) in enumerate(exp_data.items()):
        epochs = sorted(data.keys())
        
        # GIoU Loss
        ax = axes[0]
        giou_losses = [data[e].get('giou', 0) for e in epochs]
        ax.plot(epochs, giou_losses, label=exp_name, color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('GIoU Loss')
        ax.set_title('GIoU Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # L1 Loss
        ax = axes[1]
        l1_losses = [data[e].get('l1', 0) for e in epochs]
        ax.plot(epochs, l1_losses, label=exp_name, color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L1 Loss')
        ax.set_title('L1 Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Location Loss
        ax = axes[2]
        location_losses = [data[e].get('location', 0) for e in epochs]
        ax.plot(epochs, location_losses, label=exp_name, color=colors[i % len(colors)], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Location Loss')
        ax.set_title('Location Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_components.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/loss_components.png")
    plt.close()
    
    # 3. 对比柱状图（最终指标）
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Final Metrics Comparison', fontsize=14, fontweight='bold')
    
    metrics = ['loss', 'iou', 'precision', 'recall']
    titles = ['Final Loss', 'Final IoU', 'Final Precision', 'Final Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        values = []
        labels = []
        for exp_name, data in exp_data.items():
            last_epoch = max(data.keys())
            values.append(data[last_epoch].get(metric, 0))
            labels.append(exp_name)
        
        bars = ax.bar(range(len(labels)), values, color=colors[:len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel(title)
        ax.set_title(title)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/final_metrics_comparison.png")
    plt.close()


def generate_html_report(exp_data, output_dir='visualization'):
    """生成HTML报告"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SUTrack Training Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .metric-good {{ color: green; font-weight: bold; }}
        .metric-bad {{ color: red; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>SUTrack 训练结果可视化报告</h1>
        <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>实验汇总</h2>
            <p>共完成 <strong>{len(exp_data)}</strong> 个实验</p>
        </div>
        
        <h2>最终指标对比</h2>
        <table>
            <tr>
                <th>实验名称</th>
                <th>Epochs</th>
                <th>Final Loss</th>
                <th>IoU</th>
                <th>Precision</th>
                <th>Recall</th>
            </tr>
"""
    
    for exp_name, data in exp_data.items():
        last_epoch = max(data.keys())
        metrics = data[last_epoch]
        html_content += f"""
            <tr>
                <td><strong>{exp_name}</strong></td>
                <td>{last_epoch}</td>
                <td>{metrics.get('loss', 0):.4f}</td>
                <td class="metric-good">{metrics.get('iou', 0):.4f}</td>
                <td>{metrics.get('precision', 0):.4f}</td>
                <td class="metric-good">{metrics.get('recall', 0):.4f}</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h2>训练曲线</h2>
        <img src="training_curves.png" alt="Training Curves">
        
        <h2>损失分解</h2>
        <img src="loss_components.png" alt="Loss Components">
        
        <h2>最终指标对比</h2>
        <img src="final_metrics_comparison.png" alt="Final Metrics">
    </div>
</body>
</html>
"""
    
    html_path = os.path.join(output_dir, 'report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved: {html_path}")


def main():
    """主函数"""
    output_dir = 'output'
    viz_dir = 'visualization'
    
    if not os.path.exists(output_dir):
        print(f"[错误] 输出目录不存在: {output_dir}")
        return
    
    print("="*60)
    print("SUTrack 训练结果可视化分析")
    print("="*60)
    
    # 收集所有实验数据
    exp_data = {}
    exp_dirs = [d for d in glob.glob(os.path.join(output_dir, '*')) if os.path.isdir(d)]
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        log_files = glob.glob(os.path.join(exp_dir, 'logs/*.log'))
        
        for log_file in log_files:
            print(f"\n解析: {exp_name}")
            data = parse_log_file(log_file)
            if data:
                exp_data[exp_name] = data
                print(f"  [OK] 解析成功: {len(data)} epochs")
    
    if not exp_data:
        print("\n[错误] 未找到有效的训练数据")
        return
    
    # 生成可视化
    print(f"\n{'='*60}")
    print("生成可视化图表...")
    print("="*60)
    plot_training_curves(exp_data, viz_dir)
    
    # 生成HTML报告
    print(f"\n{'='*60}")
    print("生成HTML报告...")
    print("="*60)
    generate_html_report(exp_data, viz_dir)
    
    print(f"\n{'='*60}")
    print(f"完成! 所有结果保存在: {viz_dir}/")
    print(f"  - training_curves.png")
    print(f"  - loss_components.png")
    print(f"  - final_metrics_comparison.png")
    print(f"  - report.html")
    print("="*60)


if __name__ == '__main__':
    main()
