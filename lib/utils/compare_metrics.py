#!/usr/bin/env python3
"""
Comparison Experiment Data Collection Script
Collect training metrics for Standard vs Linear Attention
"""

import os
import re
import json
import argparse
from datetime import datetime


def parse_log_file(log_path):
    """Parse training log and extract key metrics"""
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
        print(f"Warning: Log file not found: {log_path}")
        return metrics
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Parse training log line
            # Format: [train: epoch, batch / total] FPS: fps (avg) , Loss/total: loss , ... IoU: iou , Precision: prec , Recall: rec
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
                
                # Only record once per epoch (avoid duplicate batches)
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


def collect_comparison_data(output_dir):
    """Collect comparison experiment data"""
    results = {}
    
    # Standard Attention
    standard_log = os.path.join(output_dir, 'b224_antiuav_standard', 'logs', 'train.log')
    if os.path.exists(standard_log):
        results['standard'] = parse_log_file(standard_log)
        print(f"[OK] Standard Attention: {len(results['standard']['epochs'])} epochs collected")
    else:
        print(f"[X] Standard Attention log not found: {standard_log}")
    
    # Linear Attention
    linear_log = os.path.join(output_dir, 'b224_antiuav_linear', 'logs', 'train.log')
    if os.path.exists(linear_log):
        results['linear'] = parse_log_file(linear_log)
        print(f"[OK] Linear Attention: {len(results['linear']['epochs'])} epochs collected")
    else:
        print(f"[X] Linear Attention log not found: {linear_log}")
    
    # Save results
    output_file = os.path.join(output_dir, 'comparison_metrics.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Data saved to: {output_file}")
    
    return results


def generate_report(results):
    """Generate comparison report"""
    print("\n" + "="*70)
    print("COMPARISON EXPERIMENT REPORT")
    print("Standard Attention vs Linear Attention")
    print("="*70)
    
    for method in ['standard', 'linear']:
        if method not in results or not results[method]['epochs']:
            continue
        
        data = results[method]
        name = "Standard Attention (O(N^2))" if method == 'standard' else "Linear Attention (O(N))"
        
        print(f"\n[{name}]")
        print(f"  Training Epochs: {max(data['epochs'])}")
        print(f"  Final Loss: {data['loss_total'][-1]:.4f}")
        print(f"  Final IoU: {data['iou'][-1]:.4f}")
        print(f"  Final Precision: {data['precision'][-1]:.4f}")
        print(f"  Final Recall: {data['recall'][-1]:.4f}")
        print(f"  Average FPS: {sum(data['fps'])/len(data['fps']):.2f}")
    
    # Calculate improvements
    if 'standard' in results and 'linear' in results:
        std_data = results['standard']
        lin_data = results['linear']
        
        if std_data['epochs'] and lin_data['epochs']:
            print("\n[PERFORMANCE COMPARISON]")
            
            # IoU comparison
            std_iou = std_data['iou'][-1]
            lin_iou = lin_data['iou'][-1]
            iou_improve = (lin_iou - std_iou) / std_iou * 100
            print(f"  IoU:      Standard={std_iou:.4f}, Linear={lin_iou:.4f}, Change={iou_improve:+.2f}%")
            
            # Speed comparison
            std_fps = sum(std_data['fps']) / len(std_data['fps'])
            lin_fps = sum(lin_data['fps']) / len(lin_data['fps'])
            fps_improve = (lin_fps - std_fps) / std_fps * 100
            print(f"  FPS:      Standard={std_fps:.2f}, Linear={lin_fps:.2f}, Change={fps_improve:+.2f}%")
            
            # Loss comparison
            std_loss = std_data['loss_total'][-1]
            lin_loss = lin_data['loss_total'][-1]
            loss_improve = (std_loss - lin_loss) / std_loss * 100
            print(f"  Loss:     Standard={std_loss:.4f}, Linear={lin_loss:.4f}, Improvement={loss_improve:+.2f}%")
            
            # Precision comparison
            std_prec = std_data['precision'][-1]
            lin_prec = lin_data['precision'][-1]
            prec_improve = (lin_prec - std_prec) / std_prec * 100
            print(f"  Precision: Standard={std_prec:.4f}, Linear={lin_prec:.4f}, Change={prec_improve:+.2f}%")
            
            # Recall comparison
            std_rec = std_data['recall'][-1]
            lin_rec = lin_data['recall'][-1]
            rec_improve = (lin_rec - std_rec) / std_rec * 100
            print(f"  Recall:   Standard={std_rec:.4f}, Linear={lin_rec:.4f}, Change={rec_improve:+.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect comparison metrics for attention mechanisms')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory containing experiment results')
    args = parser.parse_args()
    
    results = collect_comparison_data(args.output_dir)
    generate_report(results)
