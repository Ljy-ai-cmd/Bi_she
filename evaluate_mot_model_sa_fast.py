#!/usr/bin/env python3
"""
SUTrack MOT Model Evaluation — 极速版 (FP16 + 预加载 + 批量IO)
evaluate_mot_model_sa_fast.py
"""

import os, sys, json, time
import torch
import numpy as np
import cv2

prj = os.path.dirname(os.path.abspath(__file__))
if prj not in sys.path:
    sys.path.insert(0, prj)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK


def compute_iou_batch(pred_boxes, gt_boxes):
    pred_boxes = np.array(pred_boxes, dtype=np.float32)
    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    pred_x1, pred_y1 = pred_boxes[:, 0], pred_boxes[:, 1]
    pred_x2, pred_y2 = pred_boxes[:, 0] + pred_boxes[:, 2], pred_boxes[:, 1] + pred_boxes[:, 3]
    gt_x1, gt_y1 = gt_boxes[:, 0], gt_boxes[:, 1]
    gt_x2, gt_y2 = gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]
    x_left = np.maximum(pred_x1, gt_x1)
    y_top = np.maximum(pred_y1, gt_y1)
    x_right = np.minimum(pred_x2, gt_x2)
    y_bottom = np.minimum(pred_y2, gt_y2)
    inter = np.maximum(0, x_right - x_left) * np.maximum(0, y_bottom - y_top)
    area_pred = pred_boxes[:, 2] * pred_boxes[:, 3]
    area_gt = gt_boxes[:, 2] * gt_boxes[:, 3]
    union = area_pred + area_gt - inter
    return np.where(union > 0, inter / union, 0.0)


def calc_state_accuracy_fast(pred_bboxes, gt_bboxes, visibility_labels):
    pred_bboxes = np.asarray(pred_bboxes, dtype=np.float32)
    gt_bboxes = np.asarray(gt_bboxes, dtype=np.float32)
    visibility_labels = np.asarray(visibility_labels, dtype=np.float32)
    ious = compute_iou_batch(pred_bboxes, gt_bboxes)
    visible_mask = visibility_labels > 0
    pred_areas = pred_bboxes[:, 2] * pred_bboxes[:, 3]
    frame_scores = np.where(visible_mask, ious, np.where(pred_areas > 100, 1.0, 0.0))
    return np.mean(frame_scores) * 100


def load_infrared_annotations_fast(seq_dir):
    json_file = os.path.join(seq_dir, 'infrared.json')
    if not os.path.exists(json_file):
        return None, None
    with open(json_file, 'r') as f:
        data = json.load(f)
    exist = np.array(data.get('exist', []), dtype=np.float32)
    gt_rect = data.get('gt_rect', [])
    processed_gt = np.zeros((len(gt_rect), 4), dtype=np.float32)
    for i, rect in enumerate(gt_rect):
        if isinstance(rect, list) and len(rect) >= 4:
            processed_gt[i] = rect[:4]
    return exist, processed_gt


@torch.inference_mode()
def track_frame(tracker, frame_rgb, use_amp):
    with torch.cuda.amp.autocast(enabled=use_amp):
        return tracker.track(frame_rgb)


def evaluate_infrared_sequence_fast(tracker, seq_dir, gt_boxes, exist_flags, use_amp):
    video_path = os.path.join(seq_dir, 'infrared.mp4')
    if not os.path.exists(video_path):
        return None

    init_frame = int(np.argmax(exist_flags > 0)) if np.any(exist_flags > 0) else 0
    if init_frame >= len(gt_boxes):
        return None

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    total_frames = len(frames)
    max_frames = min(len(gt_boxes), total_frames) - init_frame

    init_info = {'init_bbox': gt_boxes[init_frame].tolist()}
    tracker.initialize(frames[init_frame], init_info)

    pred_boxes = np.empty((max_frames, 4), dtype=np.float32)
    pred_boxes[0] = gt_boxes[init_frame]
    track_times = np.empty(max_frames - 1, dtype=np.float32)

    for frame_idx in range(1, max_frames):
        t0 = time.perf_counter()
        outputs = track_frame(tracker, frames[init_frame + frame_idx], use_amp)
        track_times[frame_idx - 1] = time.perf_counter() - t0
        pred_boxes[frame_idx] = outputs['target_bbox']

    valid_gt = gt_boxes[init_frame:init_frame + max_frames]
    valid_exist = exist_flags[init_frame:init_frame + max_frames]

    state_accuracy = calc_state_accuracy_fast(pred_boxes, valid_gt, valid_exist)

    visible_mask = valid_exist > 0
    vis_count = int(np.sum(visible_mask))
    if vis_count > 0:
        visible_ious = compute_iou_batch(pred_boxes[visible_mask], valid_gt[visible_mask])
        thresholds = np.arange(0, 1.05, 0.05)
        auc = np.mean(np.mean(visible_ious[:, None] >= thresholds[None, :], axis=0)) * 100
        sa_05 = np.mean(visible_ious >= 0.5) * 100
        pred_c = pred_boxes[visible_mask, :2] + pred_boxes[visible_mask, 2:] / 2
        gt_c = valid_gt[visible_mask, :2] + valid_gt[visible_mask, 2:] / 2
        precision_20 = np.mean(np.sqrt(np.sum((pred_c - gt_c) ** 2, axis=1)) <= 20) * 100
    else:
        auc, sa_05, precision_20 = 0.0, 0.0, 0.0

    avg_fps = 1.0 / float(np.mean(track_times)) if max_frames > 1 else 0.0

    return {'sa': state_accuracy, 'auc': auc, 'precision': precision_20,
            'sa_05': sa_05, 'fps': avg_fps,
            'num_frames': max_frames, 'visible_frames': vis_count}


def main():
    MODEL_PATH = r'E:\biyesheji\SUTrack-main11\output\b224_antiuav_mot_v2\checkpoints\train\sutrack\sutrack_b224_antiuav_mot_v2\SUTRACK_ep0160.pth.tar'
    CONFIG_NAME = 'sutrack_b224_antiuav_mot_v2'
    TEST_DIR = r'E:\biyesheji\SUTrack-main11\data\AntI-UAV\test'
    USE_AMP = True

    print(f"MOT-v2: {CONFIG_NAME} | {os.path.basename(MODEL_PATH)} | FP16={USE_AMP}", flush=True)

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model not found: {MODEL_PATH}"); return
    if not os.path.exists(TEST_DIR):
        print(f"[Error] Test dir not found: {TEST_DIR}"); return

    params = parameters(CONFIG_NAME)
    params.checkpoint = MODEL_PATH
    params.debug = False

    sequences = []
    for seq_name in sorted(os.listdir(TEST_DIR)):
        seq_dir = os.path.join(TEST_DIR, seq_name)
        if os.path.isdir(seq_dir) and os.path.exists(os.path.join(seq_dir, 'infrared.mp4')):
            sequences.append((seq_name, seq_dir))

    all_results = {}
    total_sa, total_auc, total_prec, total_sa05, total_fps = [], [], [], [], []

    print(f"Seqs: {len(sequences)}", flush=True)
    start_time = time.time()
    tracker = SUTRACK(params, 'GOT10K')

    for i, (seq_name, seq_dir) in enumerate(sequences):
        exist, gt_boxes = load_infrared_annotations_fast(seq_dir)
        if gt_boxes is None or len(gt_boxes) == 0:
            continue

        result = evaluate_infrared_sequence_fast(tracker, seq_dir, gt_boxes, exist, USE_AMP)

        if result:
            all_results[seq_name] = result
            total_sa.append(result['sa']); total_auc.append(result['auc'])
            total_prec.append(result['precision']); total_sa05.append(result['sa_05'])
            total_fps.append(result['fps'])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(sequences)}]", flush=True)

    elapsed_time = time.time() - start_time

    if total_sa:
        avg_sa = np.mean(total_sa); avg_auc = np.mean(total_auc)
        avg_precision = np.mean(total_prec); avg_sa05 = np.mean(total_sa05)
        avg_fps = np.mean(total_fps)

        print(f"\n{'='*50}")
        print(f"SA:{avg_sa:.2f}%  AUC:{avg_auc:.2f}%  Prec:{avg_precision:.2f}%  SA@0.5:{avg_sa05:.2f}%  FPS:{avg_fps:.1f}  Time:{elapsed_time:.0f}s")
        print(f"{'='*50}")

        results = {
            'model': MODEL_PATH, 'config': CONFIG_NAME,
            'model_type': 'MOT-v2 (TTE + STCA)',
            'modality': 'infrared', 'algorithm': 'SA',
            'num_sequences': len(all_results), 'total_time': elapsed_time,
            'average': {'sa': float(avg_sa), 'auc': float(avg_auc),
                        'precision': float(avg_precision), 'sa_05': float(avg_sa05),
                        'fps': float(avg_fps)},
            'per_sequence': all_results
        }

        output_file = 'evaluation_results_mot_v2_ep0160.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {output_file}")
    else:
        print("[Error] No results")


if __name__ == '__main__':
    main()
