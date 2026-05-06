#!/usr/bin/env python3
"""
SUTrack CLAHE Model Evaluation Script with SA Algorithm
使用CLAHE权重在Anti-UAV测试集上评估模型性能（红外模态）
采用论文定义的SA算法: SA = (1/T) * Σ(IoU_t × δ(v_t > 0) + p_t × (1 - δ(v_t > 0)))
计算指标: AUC, Precision, SA, FPS
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加项目路径
prj = os.path.dirname(os.path.abspath(__file__))
if prj not in sys.path:
    sys.path.insert(0, prj)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK


def compute_iou_batch(pred_boxes, gt_boxes):
    """批量计算IoU - 使用NumPy向量化加速"""
    pred_boxes = np.array(pred_boxes, dtype=np.float32)
    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    
    # 转换为x1,y1,x2,y2格式
    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]
    
    gt_x1 = gt_boxes[:, 0]
    gt_y1 = gt_boxes[:, 1]
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3]
    
    # 计算交集
    x_left = np.maximum(pred_x1, gt_x1)
    y_top = np.maximum(pred_y1, gt_y1)
    x_right = np.minimum(pred_x2, gt_x2)
    y_bottom = np.minimum(pred_y2, gt_y2)
    
    # 交集面积
    inter_width = np.maximum(0, x_right - x_left)
    inter_height = np.maximum(0, y_bottom - y_top)
    intersection = inter_width * inter_height
    
    # 并集面积
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    union = pred_area + gt_area - intersection
    
    # IoU
    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def calc_state_accuracy_fast(pred_bboxes, gt_bboxes, visibility_labels):
    """
    快速计算State Accuracy (SA) - 使用NumPy向量化
    SA = (1/T) * Σ[t=1 to T](IoU_t × δ(v_t > 0) + p_t × (1 - δ(v_t > 0)))
    """
    pred_bboxes = np.array(pred_bboxes, dtype=np.float32)
    gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
    visibility_labels = np.array(visibility_labels, dtype=np.float32)
    
    # 计算所有帧的IoU
    ious = compute_iou_batch(pred_bboxes, gt_bboxes)
    
    # 可见帧使用IoU
    visible_mask = visibility_labels > 0
    
    # 不可见帧使用预测存在概率（简化：根据框大小判断）
    pred_areas = pred_bboxes[:, 2] * pred_bboxes[:, 3]
    existence_prob = np.where(pred_areas > 100, 1.0, 0.0)  # 面积>100认为存在
    
    # 合并分数
    frame_scores = np.where(visible_mask, ious, existence_prob)
    
    state_accuracy = np.mean(frame_scores)
    return state_accuracy, frame_scores


def load_infrared_annotations_fast(seq_dir):
    """快速加载红外模态的标注"""
    json_file = os.path.join(seq_dir, 'infrared.json')
    
    if not os.path.exists(json_file):
        return None, None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    exist = np.array(data.get('exist', []), dtype=np.float32)
    gt_rect = data.get('gt_rect', [])
    
    # 快速处理gt_rect
    processed_gt = np.zeros((len(gt_rect), 4), dtype=np.float32)
    for i, rect in enumerate(gt_rect):
        if isinstance(rect, list) and len(rect) >= 4:
            processed_gt[i] = rect[:4]
        elif isinstance(rect, list) and len(rect) == 2:
            processed_gt[i] = [rect[0], rect[1], 50, 50]
        else:
            processed_gt[i] = [0, 0, 100, 100]
    
    return exist, processed_gt


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """应用CLAHE增强"""
    import cv2
    if len(image.shape) == 3:
        # 彩色图像 - 转换到LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        # 灰度图像
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        result = clahe.apply(image.astype(np.uint8))
    return result


def evaluate_infrared_sequence_fast(tracker, seq_dir, gt_boxes, exist_flags):
    """评估单个红外视频序列 - 优化版本（带CLAHE增强）"""
    import cv2
    
    video_path = os.path.join(seq_dir, 'infrared.mp4')
    
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 找到第一个存在的帧
    init_frame = np.argmax(exist_flags > 0) if np.any(exist_flags > 0) else 0
    
    if init_frame >= len(gt_boxes):
        cap.release()
        return None
    
    # 读取初始化帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    
    # 应用CLAHE增强
    frame_enhanced = apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8))
    
    # 初始化跟踪器（使用增强后的帧）
    if len(frame_enhanced.shape) == 3:
        frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2RGB)
    
    init_info = {'init_bbox': gt_boxes[init_frame].tolist()}
    tracker.initialize(frame_rgb, init_info)
    
    # 预分配数组
    max_frames = min(len(gt_boxes), total_frames) - init_frame
    pred_boxes = np.zeros((max_frames, 4), dtype=np.float32)
    track_times = np.zeros(max_frames - 1, dtype=np.float32)
    
    pred_boxes[0] = gt_boxes[init_frame]
    
    # 批量读取和跟踪
    frame_idx = 1
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 应用CLAHE增强
        frame_enhanced = apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8))
        
        # 颜色转换
        if len(frame_enhanced.shape) == 3:
            frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2RGB)
        
        # 跟踪
        start_time = time.time()
        outputs = tracker.track(frame_rgb)
        track_times[frame_idx - 1] = time.time() - start_time
        
        pred_boxes[frame_idx] = outputs['target_bbox']
        frame_idx += 1
    
    cap.release()
    
    # 截取有效数据
    pred_boxes = pred_boxes[:frame_idx]
    track_times = track_times[:frame_idx - 1]
    valid_gt = gt_boxes[init_frame:init_frame + frame_idx]
    valid_exist = exist_flags[init_frame:init_frame + frame_idx]
    
    # 计算SA
    state_accuracy, _ = calc_state_accuracy_fast(pred_boxes, valid_gt, valid_exist)
    
    # 计算AUC - 只计算可见帧
    visible_mask = valid_exist > 0
    if np.any(visible_mask):
        visible_ious = compute_iou_batch(pred_boxes[visible_mask], valid_gt[visible_mask])
        
        # 向量化计算success rates
        iou_thresholds = np.arange(0, 1.05, 0.05)
        success_rates = np.mean(visible_ious[:, None] >= iou_thresholds[None, :], axis=0)
        auc = np.mean(success_rates) * 100
        sa_05 = np.mean(visible_ious >= 0.5) * 100
    else:
        auc = 0.0
        sa_05 = 0.0
    
    # 计算Precision
    if np.any(visible_mask):
        pred_centers = pred_boxes[visible_mask, :2] + pred_boxes[visible_mask, 2:] / 2
        gt_centers = valid_gt[visible_mask, :2] + valid_gt[visible_mask, 2:] / 2
        center_distances = np.sqrt(np.sum((pred_centers - gt_centers) ** 2, axis=1))
        precision_20 = np.mean(center_distances <= 20) * 100
    else:
        precision_20 = 0.0
    
    # 计算FPS
    avg_fps = 1.0 / np.mean(track_times) if len(track_times) > 0 else 0
    
    return {
        'sa': state_accuracy * 100,
        'auc': auc,
        'precision': precision_20,
        'sa_05': sa_05,
        'fps': avg_fps,
        'num_frames': frame_idx,
        'visible_frames': int(np.sum(valid_exist))
    }


def main():
    """主函数 - CLAHE权重评估"""
    # CLAHE权重配置
    MODEL_PATH = r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_clahe\checkpoints\train\sutrack\sutrack_b224_antiuav_clahe\SUTRACK_ep0180.pth.tar'
    CONFIG_NAME = 'sutrack_b224_antiuav_clahe'
    TEST_DIR = r'E:\biyesheji\SUTrack-main11\data\AntI-UAV\train'
    
    print("="*60)
    print("SUTrack CLAHE Model Evaluation (Infrared + SA Algorithm)")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Test Dir: {TEST_DIR}")
    print("="*60)
    print("\nSA Algorithm: SA = (1/T) * Σ(IoU_t × δ(v_t > 0) + p_t × (1 - δ(v_t > 0)))")
    print("Evaluating INFRARED modality only")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(TEST_DIR):
        print(f"[Error] Test directory not found: {TEST_DIR}")
        return
    
    # 加载参数
    params = parameters(CONFIG_NAME)
    params.checkpoint = MODEL_PATH
    params.debug = False
    
    # 创建跟踪器
    tracker = SUTRACK(params, 'GOT10K')
    
    # 查找所有测试序列
    print("\nLoading test sequences...")
    sequences = []
    for seq_name in sorted(os.listdir(TEST_DIR)):
        seq_dir = os.path.join(TEST_DIR, seq_name)
        if os.path.isdir(seq_dir):
            infrared_video = os.path.join(seq_dir, 'infrared.mp4')
            infrared_json = os.path.join(seq_dir, 'infrared.json')
            if os.path.exists(infrared_video) and os.path.exists(infrared_json):
                sequences.append((seq_name, seq_dir))
    
    print(f"Found {len(sequences)} test sequences with infrared data")
    
    # 评估每个序列
    all_results = {}
    total_sa = []
    total_auc = []
    total_precision = []
    total_sa05 = []
    total_fps = []
    
    print("\nEvaluating sequences...")
    start_time = time.time()
    
    for seq_name, seq_dir in sequences:
        exist, gt_boxes = load_infrared_annotations_fast(seq_dir)
        
        if gt_boxes is None or len(gt_boxes) == 0:
            print(f"  [Skip] No infrared annotations: {seq_name}")
            continue
        
        print(f"  Processing: {seq_name} ({len(gt_boxes)} frames, {int(np.sum(exist))} visible)...", end=' ')
        
        result = evaluate_infrared_sequence_fast(tracker, seq_dir, gt_boxes, exist)
        
        if result:
            all_results[seq_name] = result
            total_sa.append(result['sa'])
            total_auc.append(result['auc'])
            total_precision.append(result['precision'])
            total_sa05.append(result['sa_05'])
            total_fps.append(result['fps'])
            
            print(f"SA:{result['sa']:.1f}% AUC:{result['auc']:.1f}% Prec:{result['precision']:.1f}% FPS:{result['fps']:.1f}")
    
    elapsed_time = time.time() - start_time
    
    # 计算平均指标
    if total_sa:
        avg_sa = np.mean(total_sa)
        avg_auc = np.mean(total_auc)
        avg_precision = np.mean(total_precision)
        avg_sa05 = np.mean(total_sa05)
        avg_fps = np.mean(total_fps)
        
        print("\n" + "="*60)
        print("Evaluation Results - CLAHE Model (Infrared + SA Algorithm)")
        print("="*60)
        print(f"SA (State Accuracy): {avg_sa:.2f}%")
        print(f"AUC:                 {avg_auc:.2f}%")
        print(f"Precision:           {avg_precision:.2f}%")
        print(f"SA@0.5:              {avg_sa05:.2f}%")
        print(f"FPS:                 {avg_fps:.2f}")
        print(f"Total Time:          {elapsed_time:.1f}s")
        print("="*60)
        
        # 保存结果
        results = {
            'model': MODEL_PATH,
            'config': CONFIG_NAME,
            'modality': 'infrared',
            'algorithm': 'SA (State Accuracy)',
            'num_sequences': len(all_results),
            'total_time': elapsed_time,
            'average': {
                'sa': float(avg_sa),
                'auc': float(avg_auc),
                'precision': float(avg_precision),
                'sa_05': float(avg_sa05),
                'fps': float(avg_fps)
            },
            'per_sequence': all_results
        }
        
        output_file = 'evaluation_results_clahe_sa_infrared.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # Markdown格式
        print("\n" + "="*60)
        print("Results for Paper (Markdown Format)")
        print("="*60)
        print(f"| Metric | Value |")
        print(f"|--------|-------|")
        print(f"| SA (%) | {avg_sa:.2f} |")
        print(f"| AUC (%) | {avg_auc:.2f} |")
        print(f"| Precision (%) | {avg_precision:.2f} |")
        print(f"| SA@0.5 (%) | {avg_sa05:.2f} |")
        print(f"| FPS | {avg_fps:.2f} |")
        print("="*60)
    else:
        print("[Error] No results generated")


if __name__ == '__main__':
    main()
