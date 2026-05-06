#!/usr/bin/env python3
"""Server-side Standard v2 evaluation — Anti-UAV infrared"""
import os, sys, json, time
import torch
import numpy as np
import cv2

prj = '/data/gcj/wch/SUTrack-main11'
sys.path.insert(0, prj)

from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK

MODEL_PATH = f'{prj}/output/b224_antiuav_standard_v2/checkpoints/train/sutrack/sutrack_b224_antiuav_standard_v2/SUTRACK_ep0090.pth.tar'
CONFIG_NAME = 'sutrack_b224_antiuav_standard_v2'
TEST_DIR = f'{prj}/data/AntI-UAV/test'
USE_AMP = True


def compute_iou_batch(pred_boxes, gt_boxes):
    pred_boxes = np.array(pred_boxes, dtype=np.float32)
    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    px1, py1 = pred_boxes[:, 0], pred_boxes[:, 1]
    px2, py2 = pred_boxes[:, 0] + pred_boxes[:, 2], pred_boxes[:, 1] + pred_boxes[:, 3]
    gx1, gy1 = gt_boxes[:, 0], gt_boxes[:, 1]
    gx2, gy2 = gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]
    inter = np.maximum(0, np.minimum(px2, gx2) - np.maximum(px1, gx1)) * \
            np.maximum(0, np.minimum(py2, gy2) - np.maximum(py1, gy1))
    union = pred_boxes[:, 2] * pred_boxes[:, 3] + gt_boxes[:, 2] * gt_boxes[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


@torch.inference_mode()
def track_frame(tracker, frame_rgb, use_amp):
    with torch.cuda.amp.autocast(enabled=use_amp):
        return tracker.track(frame_rgb)


def evaluate_sequence(tracker, seq_dir, gt_boxes, exist_flags, use_amp):
    cap = cv2.VideoCapture(os.path.join(seq_dir, 'infrared.mp4'))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    init_frame = int(np.argmax(exist_flags > 0))
    if init_frame >= len(gt_boxes):
        return None

    max_frames = min(len(gt_boxes), len(frames)) - init_frame
    tracker.initialize(frames[init_frame], {'init_bbox': gt_boxes[init_frame].tolist()})

    pred = np.empty((max_frames, 4), dtype=np.float32)
    pred[0] = gt_boxes[init_frame]
    times = np.empty(max_frames - 1, dtype=np.float32)

    for i in range(1, max_frames):
        t0 = time.perf_counter()
        pred[i] = track_frame(tracker, frames[init_frame + i], use_amp)['target_bbox']
        times[i - 1] = time.perf_counter() - t0

    valid_gt = gt_boxes[init_frame:init_frame + max_frames]
    valid_exist = exist_flags[init_frame:init_frame + max_frames]

    ious = compute_iou_batch(pred, valid_gt)
    vis = valid_exist > 0
    vis_count = int(np.sum(vis))

    if vis_count > 0:
        auc = np.mean(np.mean(ious[vis][:, None] >= np.arange(0, 1.05, 0.05)[None, :], axis=0)) * 100
        sa_05 = np.mean(ious[vis] >= 0.5) * 100
        pc = pred[vis, :2] + pred[vis, 2:] / 2
        gc = valid_gt[vis, :2] + valid_gt[vis, 2:] / 2
        prec = np.mean(np.sqrt(np.sum((pc - gc) ** 2, 1)) <= 20) * 100
    else:
        auc = sa_05 = prec = 0.0

    areas = pred[:, 2] * pred[:, 3]
    frame_scores = np.where(vis, ious, np.where(areas > 100, 1.0, 0.0))
    sa = np.mean(frame_scores) * 100

    return {'sa': sa, 'auc': auc, 'precision': prec, 'sa_05': sa_05,
            'fps': 1.0 / float(np.mean(times)) if max_frames > 1 else 0}


def main():
    print(f"Standard-v2: {os.path.basename(MODEL_PATH)} | FP16={USE_AMP}", flush=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Not found: {MODEL_PATH}"); return

    params = parameters(CONFIG_NAME)
    params.checkpoint = MODEL_PATH
    params.debug = False

    seqs = sorted([(d, os.path.join(TEST_DIR, d)) for d in os.listdir(TEST_DIR)
                   if os.path.isdir(os.path.join(TEST_DIR, d))
                   and os.path.exists(os.path.join(TEST_DIR, d, 'infrared.mp4'))])

    all_sa, all_auc, all_prec, all_sa05, all_fps = [], [], [], [], []
    results = {}
    print(f"Seqs: {len(seqs)}", flush=True)
    t0 = time.time()
    tracker = SUTRACK(params, 'GOT10K')

    for i, (name, d) in enumerate(seqs):
        with open(os.path.join(d, 'infrared.json')) as f:
            data = json.load(f)
        exist = np.array(data['exist'], dtype=np.float32)
        gt = np.array([r[:4] for r in data.get('gt_rect', []) if isinstance(r, list) and len(r) >= 4], dtype=np.float32)

        r = evaluate_sequence(tracker, d, gt, exist, USE_AMP)
        if r:
            results[name] = r
            all_sa.append(r['sa']); all_auc.append(r['auc'])
            all_prec.append(r['precision']); all_sa05.append(r['sa_05']); all_fps.append(r['fps'])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(seqs)}]", flush=True)

    t = time.time() - t0
    if all_sa:
        sa = np.mean(all_sa); auc = np.mean(all_auc)
        prec = np.mean(all_prec); sa05 = np.mean(all_sa05); fps = np.mean(all_fps)
        print(f"\n{'='*50}")
        print(f"SA:{sa:.2f}%  AUC:{auc:.2f}%  Prec:{prec:.2f}%  SA@0.5:{sa05:.2f}%  FPS:{fps:.1f}  Time:{t:.0f}s")
        print(f"{'='*50}")

        out = {'model': MODEL_PATH, 'config': CONFIG_NAME, 'model_type': 'Standard-v2',
               'modality': 'infrared', 'algorithm': 'SA', 'num_sequences': len(seqs), 'total_time': t,
               'average': {'sa': float(sa), 'auc': float(auc), 'precision': float(prec),
                           'sa_05': float(sa05), 'fps': float(fps)}, 'per_sequence': results}
        with open(f'{prj}/evaluation_results_standard_v2_ep0090.json', 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved: evaluation_results_standard_v2_ep0090.json")


if __name__ == '__main__':
    main()
