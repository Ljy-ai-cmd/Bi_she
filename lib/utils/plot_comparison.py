#!/usr/bin/env python3
"""
Comparison Experiment Visualization Script
Generate figures for paper writing
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_convergence_curves(data, output_path):
    """Plot convergence curves comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    methods = [
        ('standard', 'Standard Attention', 'blue', 'O(N^2)'),
        ('linear', 'Linear Attention', 'red', 'O(N)')
    ]
    
    # Loss curves
    ax = axes[0, 0]
    for method, label, color, complexity in methods:
        if method in data and data[method]['epochs']:
            ax.plot(data[method]['epochs'], data[method]['loss_total'],
                   label=f'{label} ({complexity})', color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('(a) Training Loss Convergence', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # IoU curves
    ax = axes[0, 1]
    for method, label, color, complexity in methods:
        if method in data and data[method]['epochs']:
            ax.plot(data[method]['epochs'], data[method]['iou'],
                   label=f'{label} ({complexity})', color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('(b) IoU Improvement', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Precision curves
    ax = axes[0, 2]
    for method, label, color, complexity in methods:
        if method in data and data[method]['epochs']:
            ax.plot(data[method]['epochs'], data[method]['precision'],
                   label=f'{label} ({complexity})', color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('(c) Precision Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Recall curves
    ax = axes[1, 0]
    for method, label, color, complexity in methods:
        if method in data and data[method]['epochs']:
            ax.plot(data[method]['epochs'], data[method]['recall'],
                   label=f'{label} ({complexity})', color=color, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('(d) Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # FPS comparison
    ax = axes[1, 1]
    method_names = []
    fps_means = []
    colors = []
    for method, label, color, complexity in methods:
        if method in data and data[method]['fps']:
            method_names.append(f'{label}\n({complexity})')
            fps_means.append(np.mean(data[method]['fps']))
            colors.append(color)
    
    if method_names:
        bars = ax.bar(method_names, fps_means, color=colors, alpha=0.7)
        ax.set_ylabel('FPS (Frames Per Second)', fontsize=12)
        ax.set_title('(e) Training Speed Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, fps in zip(bars, fps_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{fps:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Final metrics comparison
    ax = axes[1, 2]
    metrics_names = ['IoU', 'Precision', 'Recall']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    std_values = []
    lin_values = []
    
    if 'standard' in data and data['standard']['epochs']:
        std_values = [
            data['standard']['iou'][-1],
            data['standard']['precision'][-1],
            data['standard']['recall'][-1]
        ]
    
    if 'linear' in data and data['linear']['epochs']:
        lin_values = [
            data['linear']['iou'][-1],
            data['linear']['precision'][-1],
            data['linear']['recall'][-1]
        ]
    
    if std_values and lin_values:
        bars1 = ax.bar(x - width/2, std_values, width, label='Standard O(N^2)', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, lin_values, width, label='Linear O(N)', color='red', alpha=0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('(f) Final Performance Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] Figure saved to: {output_path}")
    plt.close()


def generate_latex_table(data, output_path):
    """Generate LaTeX table for paper"""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison between Standard Attention and Linear Attention on Anti-UAV Dataset}")
    latex.append("\\label{tab:attention_comparison}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("Method & Complexity & Final Loss & IoU ($\\uparrow$) & Precision ($\\uparrow$) & Recall ($\\uparrow$) & FPS ($\\uparrow$) \\\\")
    latex.append("\\midrule")
    
    for method, name, complexity in [('standard', 'Standard Attention', '$O(N^2)$'),
                                      ('linear', 'Linear Attention', '$O(N)$')]:
        if method in data and data[method]['epochs']:
            loss = data[method]['loss_total'][-1]
            iou = data[method]['iou'][-1]
            prec = data[method]['precision'][-1]
            rec = data[method]['recall'][-1]
            fps = np.mean(data[method]['fps'])
            latex.append(f"{name} & {complexity} & {loss:.4f} & {iou:.4f} & {prec:.4f} & {rec:.4f} & {fps:.1f} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"[Saved] LaTeX table saved to: {output_path}")


def generate_markdown_report(data, output_path):
    """Generate Markdown report"""
    md = []
    md.append("# Attention Mechanism Comparison Report")
    md.append("\n## Experiment Setup")
    md.append("- Dataset: Anti-UAV")
    md.append("- Model: Fast-iTPN Base (b224)")
    md.append("- Input: RGBT Fusion (6 channels)")
    md.append("- Training Epochs: 180")
    md.append("\n## Results")
    md.append("\n| Method | Complexity | Final Loss | IoU | Precision | Recall | FPS |")
    md.append("|--------|-----------|------------|-----|-----------|--------|-----|")
    
    for method, name, complexity in [('standard', 'Standard Attention', 'O(N^2)'),
                                      ('linear', 'Linear Attention', 'O(N)')]:
        if method in data and data[method]['epochs']:
            loss = data[method]['loss_total'][-1]
            iou = data[method]['iou'][-1]
            prec = data[method]['precision'][-1]
            rec = data[method]['recall'][-1]
            fps = np.mean(data[method]['fps'])
            md.append(f"| {name} | {complexity} | {loss:.4f} | {iou:.4f} | {prec:.4f} | {rec:.4f} | {fps:.1f} |")
    
    # Add comparison
    if 'standard' in data and 'linear' in data:
        std_data = data['standard']
        lin_data = data['linear']
        
        if std_data['epochs'] and lin_data['epochs']:
            md.append("\n## Performance Improvement")
            
            std_fps = np.mean(std_data['fps'])
            lin_fps = np.mean(lin_data['fps'])
            fps_gain = (lin_fps - std_fps) / std_fps * 100
            md.append(f"\n- **Speed Improvement**: {fps_gain:+.2f}% ({std_fps:.1f} → {lin_fps:.1f} FPS)")
            
            std_iou = std_data['iou'][-1]
            lin_iou = lin_data['iou'][-1]
            iou_gain = (lin_iou - std_iou) / std_iou * 100
            md.append(f"- **IoU Change**: {iou_gain:+.2f}% ({std_iou:.4f} → {lin_iou:.4f})")
            
            std_loss = std_data['loss_total'][-1]
            lin_loss = lin_data['loss_total'][-1]
            loss_gain = (std_loss - lin_loss) / std_loss * 100
            md.append(f"- **Loss Reduction**: {loss_gain:+.2f}% ({std_loss:.4f} → {lin_loss:.4f})")
    
    md.append("\n## Conclusion")
    md.append("Linear attention achieves comparable tracking accuracy with significantly improved computational efficiency.")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"[Saved] Markdown report saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot comparison figures for paper')
    parser.add_argument('--json', type=str, default='./output/comparison_metrics.json',
                        help='Path to comparison metrics JSON')
    parser.add_argument('--output_dir', type=str, default='./output/comparison_figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found: {args.json}")
        print("Please run compare_metrics.py first to collect data.")
        exit(1)
    
    with open(args.json, 'r') as f:
        data = json.load(f)
    
    # Generate figures
    plot_convergence_curves(data, os.path.join(args.output_dir, 'convergence_comparison.png'))
    
    # Generate LaTeX table
    generate_latex_table(data, os.path.join(args.output_dir, 'comparison_table.tex'))
    
    # Generate Markdown report
    generate_markdown_report(data, os.path.join(args.output_dir, 'comparison_report.md'))
    
    print(f"\n[Done] All results saved to: {args.output_dir}")
