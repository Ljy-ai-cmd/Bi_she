#!/usr/bin/env python3
"""
CLAHE Ablation Study Data Collection
CLAHE预处理消融实验数据收集脚本
"""

import os
import re
import json
import argparse
import numpy as np


def parse_log_file(log_path):
    """解析训练日志"""
    metrics = {
        'epochs': [],
        'loss_total': [],
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
                r'.*IoU:\s+([\d.]+)'
                r'.*Precision:\s+([\d.]+)'
                r'.*Recall:\s+([\d.]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                fps = float(match.group(2))
                loss = float(match.group(3))
                iou = float(match.group(4))
                precision = float(match.group(5))
                recall = float(match.group(6))
                
                if epoch not in metrics['epochs']:
                    metrics['epochs'].append(epoch)
                    metrics['loss_total'].append(loss)
                    metrics['iou'].append(iou)
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['fps'].append(fps)
    
    return metrics


def collect_clahe_comparison(output_dir):
    """收集CLAHE消融实验数据"""
    results = {}
    
    experiments = {
        'baseline': ('b224_antiuav_linear', 'Baseline (No CLAHE)'),
        'clahe': ('b224_antiuav_clahe', 'With CLAHE'),
        'standard': ('b224_antiuav_standard', 'Standard Attention')
    }
    
    for key, (exp_dir, name) in experiments.items():
        log_path = os.path.join(output_dir, exp_dir, 'logs', 'train.log')
        if os.path.exists(log_path):
            results[key] = parse_log_file(log_path)
            print(f"[OK] {name}: {len(results[key]['epochs'])} epochs")
        else:
            print(f"[X] {name}: Log not found")
    
    # Save results
    output_file = os.path.join(output_dir, 'clahe_comparison.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Data saved to: {output_file}")
    
    return results


def generate_clahe_report(results):
    """生成CLAHE消融实验报告"""
    print("\n" + "="*70)
    print("CLAHE ABLATION STUDY REPORT")
    print("="*70)
    
    for key in ['baseline', 'clahe', 'standard']:
        if key not in results or not results[key]['epochs']:
            continue
        
        data = results[key]
        name = {
            'baseline': 'Baseline (Linear, No CLAHE)',
            'clahe': 'Linear + CLAHE',
            'standard': 'Standard Attention'
        }.get(key, key)
        
        print(f"\n[{name}]")
        print(f"  Epochs: {max(data['epochs'])}")
        print(f"  Final Loss: {data['loss_total'][-1]:.4f}")
        print(f"  Final IoU: {data['iou'][-1]:.4f}")
        print(f"  Final Precision: {data['precision'][-1]:.4f}")
        print(f"  Final Recall: {data['recall'][-1]:.4f}")
        print(f"  Average FPS: {np.mean(data['fps']):.2f}")
    
    # CLAHE effect analysis
    if 'baseline' in results and 'clahe' in results:
        baseline_data = results['baseline']
        clahe_data = results['clahe']
        
        if baseline_data['epochs'] and clahe_data['epochs']:
            print("\n[CLAHE EFFECT ANALYSIS]")
            
            # IoU improvement
            base_iou = baseline_data['iou'][-1]
            clahe_iou = clahe_data['iou'][-1]
            iou_gain = (clahe_iou - base_iou) / base_iou * 100
            print(f"  IoU: {base_iou:.4f} → {clahe_iou:.4f} ({iou_gain:+.2f}%)")
            
            # Precision improvement
            base_prec = baseline_data['precision'][-1]
            clahe_prec = clahe_data['precision'][-1]
            prec_gain = (clahe_prec - base_prec) / base_prec * 100
            print(f"  Precision: {base_prec:.4f} → {clahe_prec:.4f} ({prec_gain:+.2f}%)")
            
            # Recall improvement
            base_rec = baseline_data['recall'][-1]
            clahe_rec = clahe_data['recall'][-1]
            rec_gain = (clahe_rec - base_rec) / base_rec * 100
            print(f"  Recall: {base_rec:.4f} → {clahe_rec:.4f} ({rec_gain:+.2f}%)")
            
            # Loss reduction
            base_loss = baseline_data['loss_total'][-1]
            clahe_loss = clahe_data['loss_total'][-1]
            loss_gain = (base_loss - clahe_loss) / base_loss * 100
            print(f"  Loss: {base_loss:.4f} → {clahe_loss:.4f} ({loss_gain:+.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect CLAHE ablation study data')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory containing experiment results')
    args = parser.parse_args()
    
    results = collect_clahe_comparison(args.output_dir)
    generate_clahe_report(results)
