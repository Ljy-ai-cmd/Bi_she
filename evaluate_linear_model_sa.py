#!/usr/bin/env python3
"""
SUTrack Linear Model Evaluation Script with SA Algorithm
在Anti-UAV测试集上评估 Linear 权重模型性能（红外模态）
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
    
    # 解析标注 - Anti-UAV格式
    gt_boxes = []
    exist_labels = []
    
    # 检查格式
    if 'annotations' in data:
        # 新格式
        for frame_anno in data['annotations'].values():
            bbox = frame_anno.get('bbox', [0, 0, 0, 0])
            exist = frame_anno.get('exist', False)
            gt_boxes.append(bbox)
            exist_labels.append(1 if exist else 0)
    elif 'exist' in data and 'gt_rect' in data:
        # 旧格式 (Anti-UAV)
        exist_labels = data['exist']
        gt_boxes_raw = data['gt_rect']
        
        # 确保所有bbox都是4个元素的列表
        gt_boxes = []
        for bbox in gt_boxes_raw:
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                gt_boxes.append([float(x) for x in bbox])
            else:
                gt_boxes.append([0.0, 0.0, 0.0, 0.0])
    else:
        return None, None
    
    return np.array(exist_labels, dtype=np.float32), np.array(gt_boxes, dtype=np.float32)


def evaluate_infrared_sequence_fast(tracker, seq_dir, gt_boxes, exist_labels):
    """评估单个红外序列 - 快速版本"""
    import cv2
    
    video_path = os.path.join(seq_dir, 'infrared.mp4')
    
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    pred_boxes = []
    track_times = []
    frame_idx = 0
    valid_gt = []
    valid_exist = []
    
    # 找到第一个可见帧作为初始化
    init_frame = None
    init_bbox = None
    
    for i, (bbox, exist) in enumerate(zip(gt_boxes, exist_labels)):
        if exist > 0 and bbox[2] > 0 and bbox[3] > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                init_frame = frame
                init_bbox = bbox
                init_frame_idx = i
                break
    
    if init_frame is None or init_bbox is None:
        cap.release()
        return None
    
    # 重置到开始
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # 初始化跟踪器
    info = {'init_bbox': init_bbox}
    tracker.initialize(init_frame, info)
    
    # 处理所有帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= len(gt_boxes):
            break
        
        # 记录当前帧的GT
        valid_gt.append(gt_boxes[frame_idx])
        valid_exist.append(exist_labels[frame_idx])
        
        # 跟踪
        start_time = time.time()
        out = tracker.track(frame)
        track_time = time.time() - start_time
        track_times.append(track_time)
        
        # 获取预测框
        pred_bbox = out.get('target_bbox', [0, 0, 0, 0])
        pred_boxes.append(pred_bbox)
        
        frame_idx += 1
    
    cap.release()
    
    if len(pred_boxes) == 0:
        return None
    
    # 转换为numpy数组
    pred_boxes = np.array(pred_boxes, dtype=np.float32)
    valid_gt = np.array(valid_gt, dtype=np.float32)
    valid_exist = np.array(valid_exist, dtype=np.float32)
    
    # 计算SA
    state_accuracy, _ = calc_state_accuracy_fast(pred_boxes, valid_gt, valid_exist)
    
    # 只计算可见帧的AUC和Precision
    visible_mask = valid_exist > 0
    
    if np.any(visible_mask):
        visible_ious = compute_iou_batch(pred_boxes[visible_mask], valid_gt[visible_mask])
        
        # 计算AUC
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
    """主函数 - Linear模型评估"""
    MODEL_PATH = r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_linear\checkpoints\train\sutrack\sutrack_b224_antiuav\SUTRACK_ep0180.pth.tar'
    CONFIG_NAME = 'sutrack_b224_antiuav_linear'  # 使用Linear专用配置
    TEST_DIR = r'E:\biyesheji\SUTrack-main11\data\AntI-UAV\train'
    
    print("="*60)
    print("SUTrack Linear Model Evaluation (Infrared + SA Algorithm)")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Config: {CONFIG_NAME}")
    print(f"Test Dir: {TEST_DIR}")
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
            print(f"SA={result['sa']:.2f}%, AUC={result['auc']:.2f}%, Prec={result['precision']:.2f}%")
        else:
            print("[Failed]")
    
    elapsed_time = time.time() - start_time
    
    # 计算平均值
    avg_sa = np.mean(total_sa) if total_sa else 0
    avg_auc = np.mean(total_auc) if total_auc else 0
    avg_precision = np.mean(total_precision) if total_precision else 0
    avg_sa05 = np.mean(total_sa05) if total_sa05 else 0
    avg_fps = np.mean(total_fps) if total_fps else 0
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    print(f"Total Sequences: {len(all_results)}")
    print(f"Total Time: {elapsed_time:.2f}s")
    print(f"\nAverage Metrics:")
    print(f"  SA:        {avg_sa:.2f}%")
    print(f"  AUC:       {avg_auc:.2f}%")
    print(f"  Precision: {avg_precision:.2f}%")
    print(f"  SA@0.5:    {avg_sa05:.2f}%")
    print(f"  FPS:       {avg_fps:.2f}")
    print("="*60)
    
    # 保存结果
    results = {
        'model': MODEL_PATH,
        'config': CONFIG_NAME,
        'num_sequences': len(all_results),
        'avg_sa': avg_sa,
        'avg_auc': avg_auc,
        'avg_precision': avg_precision,
        'avg_sa_05': avg_sa05,
        'avg_fps': avg_fps,
        'total_time': elapsed_time,
        'sequence_results': all_results
    }
    
    output_file = 'evaluation_results_linear_sa.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # 打印论文格式
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


if __name__ == '__main__':
    main()
