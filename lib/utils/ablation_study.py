#!/usr/bin/env python3
"""
Ablation Study for Attention Mechanisms
消融实验：对比不同注意力机制的效果

对比方案：
1. Standard Attention (O(N^2))
2. Linear Attention (O(N))
3. Selective Linear Attention (O(N) + LaSt-ViT inspired)
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    """解析训练日志"""
    metrics = {
        'epochs': [],
        'loss_total': [],
        'loss_giou': [],
        'loss_l1': [],
        'loss_location': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'fps': []
    }
    
    if not os.path.exists(log_path):
        return metrics
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(
                r'\[train:\s+(\d+),\s+\d+\s+/\s+\d+\]\s+FPS:\s+[\d.]+?\s+\(([\d.]+)\)'
                r'.*Loss/total:\s+([\d.]+)'
                r'.*Loss/giou:\s+([\d.]+)'
                r'.*Loss/l1:\s+([\d.]+)'
                r'.*Loss/location:\s+([\d.]+)'
                r'.*IoU:\s+([\d.]+)'
                r'.*Precision:\s+([\d.]+)'
                r'.*Recall:\s+([\d.]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                fps = float(match.group(2))
                loss_total = float(match.group(3))
                loss_giou = float(match.group(4))
                loss_l1 = float(match.group(5))
                loss_location = float(match.group(6))
                iou = float(match.group(7))
                precision = float(match.group(8))
                recall = float(match.group(9))
                
                if epoch not in metrics['epochs']:
                    metrics['epochs'].append(epoch)
                    metrics['loss_total'].append(loss_total)
                    metrics['loss_giou'].append(loss_giou)
                    metrics['loss_l1'].append(loss_l1)
                    metrics['loss_location'].append(loss_location)
                    metrics['iou'].append(iou)
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['fps'].append(fps)
    
    return metrics


def collect_ablation_data(output_dir):
    """收集消融实验数据"""
    results = {}
    
    experiments = {
        'standard': ('b224_antiuav_standard', 'Standard Attention O(N^2)'),
        'linear': ('b224_antiuav_linear', 'Linear Attention O(N)'),
        'selective': ('b224_antiuav_selective', 'Selective Linear Attention O(N)'),
        'clahe': ('b224_antiuav_clahe', 'Linear + CLAHE')
    }
    
    for key, (exp_dir, name) in experiments.items():
        log_path = os.path.join(output_dir, exp_dir, 'logs', 'train.log')
        if os.path.exists(log_path):
            results[key] = parse_log_file(log_path)
            print(f"[OK] {name}: {len(results[key]['epochs'])} epochs")
        else:
            print(f"[X] {name}: Log not found - {log_path}")
    
    # Save results
    output_file = os.path.join(output_dir, 'ablation_study.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Data saved to: {output_file}")
    
    return results


def generate_ablation_report(results):
    """生成消融实验报告"""
    print("\n" + "="*80)
    print("ABLATION STUDY REPORT")
    print("Attention Mechanism Comparison")
    print("="*80)
    
    method_names = {
        'standard': 'Standard Attention O(N^2)',
        'linear': 'Linear Attention O(N)',
        'selective': 'Selective Linear Attention O(N)',
        'clahe': 'Linear + CLAHE'
    }
    
    for key in ['standard', 'linear', 'selective', 'clahe']:
        if key not in results or not results[key]['epochs']:
            continue
        
        data = results[key]
        name = method_names.get(key, key)
        
        print(f"\n[{name}]")
        print(f"  Epochs: {max(data['epochs'])}")
        print(f"  Final Loss: {data['loss_total'][-1]:.4f}")
        print(f"  Final IoU: {data['iou'][-1]:.4f}")
        print(f"  Final Precision: {data['precision'][-1]:.4f}")
        print(f"  Final Recall: {data['recall'][-1]:.4f}")
        print(f"  Average FPS: {np.mean(data['fps']):.2f}")
    
    # Performance comparison
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    
    if 'standard' in results and 'linear' in results:
        std_data = results['standard']
        lin_data = results['linear']
        
        if std_data['epochs'] and lin_data['epochs']:
            print("\n[1. Linear vs Standard Attention]")
            
            # Speed
            std_fps = np.mean(std_data['fps'])
            lin_fps = np.mean(lin_data['fps'])
            fps_gain = (lin_fps - std_fps) / std_fps * 100
            print(f"  Speed: {std_fps:.2f} → {lin_fps:.2f} FPS ({fps_gain:+.2f}%)")
            
            # IoU
            std_iou = std_data['iou'][-1]
            lin_iou = lin_data['iou'][-1]
            iou_gain = (lin_iou - std_iou) / std_iou * 100
            print(f"  IoU: {std_iou:.4f} → {lin_iou:.4f} ({iou_gain:+.2f}%)")
            
            # Loss
            std_loss = std_data['loss_total'][-1]
            lin_loss = lin_data['loss_total'][-1]
            loss_gain = (std_loss - lin_loss) / std_loss * 100
            print(f"  Loss: {std_loss:.4f} → {lin_loss:.4f} ({loss_gain:+.2f}%)")
    
    if 'linear' in results and 'selective' in results:
        lin_data = results['linear']
        sel_data = results['selective']
        
        if lin_data['epochs'] and sel_data['epochs']:
            print("\n[2. Selective vs Linear Attention]")
            
            lin_fps = np.mean(lin_data['fps'])
            sel_fps = np.mean(sel_data['fps'])
            fps_gain = (sel_fps - lin_fps) / lin_fps * 100
            print(f"  Speed: {lin_fps:.2f} → {sel_fps:.2f} FPS ({fps_gain:+.2f}%)")
            
            lin_iou = lin_data['iou'][-1]
            sel_iou = sel_data['iou'][-1]
            iou_gain = (sel_iou - lin_iou) / lin_iou * 100
            print(f"  IoU: {lin_iou:.4f} → {sel_iou:.4f} ({iou_gain:+.2f}%)")
            
            lin_loss = lin_data['loss_total'][-1]
            sel_loss = sel_data['loss_total'][-1]
            loss_gain = (lin_loss - sel_loss) / lin_loss * 100
            print(f"  Loss: {lin_loss:.4f} → {sel_loss:.4f} ({loss_gain:+.2f}%)")
    
    if 'linear' in results and 'clahe' in results:
        lin_data = results['linear']
        clahe_data = results['clahe']
        
        if lin_data['epochs'] and clahe_data['epochs']:
            print("\n[3. CLAHE Enhancement Effect]")
            
            lin_iou = lin_data['iou'][-1]
            clahe_iou = clahe_data['iou'][-1]
            iou_gain = (clahe_iou - lin_iou) / lin_iou * 100
            print(f"  IoU: {lin_iou:.4f} → {clahe_iou:.4f} ({iou_gain:+.2f}%)")
            
            lin_prec = lin_data['precision'][-1]
            clahe_prec = clahe_data['precision'][-1]
            prec_gain = (clahe_prec - lin_prec) / lin_prec * 100
            print(f"  Precision: {lin_prec:.4f} → {clahe_prec:.4f} ({prec_gain:+.2f}%)")


def plot_ablation_curves(results, output_dir):
    """绘制消融实验曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = [
        ('standard', 'Standard O(N^2)', 'blue'),
        ('linear', 'Linear O(N)', 'red'),
        ('selective', 'Selective O(N)', 'green'),
        ('clahe', 'Linear+CLAHE', 'orange')
    ]
    
    # Loss curves
    ax = axes[0, 0]
    for key, label, color in methods:
        if key in results and results[key]['epochs']:
            ax.plot(results[key]['epochs'], results[key]['loss_total'],
                   label=label, color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('(a) Training Loss Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # IoU curves
    ax = axes[0, 1]
    for key, label, color in methods:
        if key in results and results[key]['epochs']:
            ax.plot(results[key]['epochs'], results[key]['iou'],
                   label=label, color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('(b) IoU Improvement', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Precision curves
    ax = axes[1, 0]
    for key, label, color in methods:
        if key in results and results[key]['epochs']:
            ax.plot(results[key]['epochs'], results[key]['precision'],
                   label=label, color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('(c) Precision Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # FPS comparison
    ax = axes[1, 1]
    method_names = []
    fps_means = []
    colors = []
    
    for key, label, color in methods:
        if key in results and results[key]['fps']:
            method_names.append(label.replace(' ', '\n'))
            fps_means.append(np.mean(results[key]['fps']))
            colors.append(color)
    
    if method_names:
        bars = ax.bar(method_names, fps_means, color=colors, alpha=0.7)
        ax.set_ylabel('FPS (Frames Per Second)', fontsize=12)
        ax.set_title('(d) Training Speed Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, fps in zip(bars, fps_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{fps:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Figure saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study for attention mechanisms')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory containing experiment results')
    args = parser.parse_args()
    
    results = collect_ablation_data(args.output_dir)
    generate_ablation_report(results)
    plot_ablation_curves(results, args.output_dir)
