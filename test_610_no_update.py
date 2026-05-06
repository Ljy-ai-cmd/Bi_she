#!/usr/bin/env python3
"""Quick test: MOT-v2 on 193610 seqs only, with template update DISABLED"""
import os, sys, json, time
import torch, numpy as np, cv2

prj = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, prj)
from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK

MODEL = f'{prj}/output/b224_antiuav_mot_v2/checkpoints/train/sutrack/sutrack_b224_antiuav_mot_v2/SUTRACK_ep0160.pth.tar'
CONFIG = 'sutrack_b224_antiuav_mot_v2'
TEST_DIR = f'{prj}/data/AntI-UAV/test'
PREFIX = '20190925_193610'
USE_AMP = True

def iou_batch(p, g):
    p, g = np.asarray(p, np.float32), np.asarray(g, np.float32)
    px1, py1 = p[:,0], p[:,1]; px2, py2 = px1+p[:,2], py1+p[:,3]
    gx1, gy1 = g[:,0], g[:,1]; gx2, gy2 = gx1+g[:,2], gy1+g[:,3]
    inter = np.maximum(0, np.minimum(px2,gx2)-np.maximum(px1,gx1)) * \
            np.maximum(0, np.minimum(py2,gy2)-np.maximum(py1,gy1))
    union = p[:,2]*p[:,3] + g[:,2]*g[:,3] - inter
    return np.where(union>0, inter/union, 0)

@torch.inference_mode()
def track(t, f, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        return t.track(f)

def eval_seq(tracker, seq_dir, gt, exist):
    cap = cv2.VideoCapture(os.path.join(seq_dir, 'infrared.mp4'))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if len(f.shape)==3 else cv2.cvtColor(f, cv2.COLOR_GRAY2RGB))
    cap.release()
    if not frames: return None

    init = int(np.argmax(exist>0))
    n = min(len(gt), len(frames)) - init
    if n<=0: return None

    tracker.initialize(frames[init], {'init_bbox': gt[init].tolist()})
    pred = np.empty((n, 4), np.float32); pred[0] = gt[init]

    for i in range(1, n):
        pred[i] = track(tracker, frames[init+i], USE_AMP)['target_bbox']

    vgt, vex = gt[init:init+n], exist[init:init+n]
    ious = iou_batch(pred, vgt)
    vis = vex>0; vc = int(vis.sum())

    if vc:
        auc = np.mean(np.mean(ious[vis,None]>=np.arange(0,1.05,0.05)[None,:],0))*100
        sa_05 = np.mean(ious[vis]>=0.5)*100
        pc, gc = pred[vis,:2]+pred[vis,2:]/2, vgt[vis,:2]+vgt[vis,2:]/2
        prec = np.mean(np.sqrt(((pc-gc)**2).sum(1))<=20)*100
    else:
        auc = sa_05 = prec = 0.0
    areas = pred[:,2]*pred[:,3]
    sa = np.mean(np.where(vis, ious, np.where(areas>100,1,0)))*100
    return {'sa': sa, 'n': n}

print(f"Template update DISABLED | Testing {PREFIX}* only", flush=True)
params = parameters(CONFIG); params.checkpoint = MODEL; params.debug = False
tracker = SUTRACK(params, 'GOT10K')

seqs = sorted([(d, os.path.join(TEST_DIR, d)) for d in os.listdir(TEST_DIR)
               if d.startswith(PREFIX) and os.path.isdir(os.path.join(TEST_DIR, d))
               and os.path.exists(os.path.join(TEST_DIR, d, 'infrared.mp4'))])

results = {}
for name, d in seqs:
    with open(os.path.join(d, 'infrared.json')) as f:
        data = json.load(f)
    exist = np.array(data['exist'], np.float32)
    gt = np.array([r[:4] for r in data.get('gt_rect',[]) if isinstance(r,list) and len(r)>=4], np.float32)
    r = eval_seq(tracker, d, gt, exist)
    if r: results[name] = r
    print(f"  {name}: SA={r['sa']:.1f}% (no update)" if r else f"  {name}: FAILED")

if results:
    sa_list = [v['sa'] for v in results.values()]
    print(f"\n193610 avg SA (no template update): {np.mean(sa_list):.2f}%")
    print(f"Compare: Standard=63.99%  |  MOT-v2 (with update)=17.30%")
