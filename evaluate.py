#!/usr/bin/env python3
"""通用 Anti-UAV 评估脚本 — 极速版 (FP16 + 预加载)
用法:
  python evaluate.py sutrack_b224_antiuav_rti SUTRACK_ep0090.pth.tar
  python evaluate.py sutrack_b224_antiuav_mot_v2 SUTRACK_ep0160.pth.tar
"""
import os, sys, json, time, argparse
import torch, numpy as np, cv2

prj = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, prj)
from lib.test.parameter.sutrack import parameters
from lib.test.tracker.sutrack import SUTRACK

USE_AMP = True


def iou_batch(pred, gt):
    pred, gt = np.asarray(pred, np.float32), np.asarray(gt, np.float32)
    px1, py1 = pred[:,0], pred[:,1]; px2, py2 = px1+pred[:,2], py1+pred[:,3]
    gx1, gy1 = gt[:,0], gt[:,1]; gx2, gy2 = gx1+gt[:,2], gy1+gt[:,3]
    inter = np.maximum(0, np.minimum(px2,gx2)-np.maximum(px1,gx1)) * \
            np.maximum(0, np.minimum(py2,gy2)-np.maximum(py1,gy1))
    union = pred[:,2]*pred[:,3] + gt[:,2]*gt[:,3] - inter
    return np.where(union > 0, inter/union, 0.0)


@torch.inference_mode()
def track(tracker, frame, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        return tracker.track(frame)


def eval_seq(tracker, seq_dir, gt, exist, amp):
    cap = cv2.VideoCapture(os.path.join(seq_dir, 'infrared.mp4'))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        if len(f.shape) == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(f)
    cap.release()
    if not frames: return None

    init = int(np.argmax(exist > 0))
    n = min(len(gt), len(frames)) - init
    if n <= 0: return None

    tracker.initialize(frames[init], {'init_bbox': gt[init].tolist()})
    pred = np.empty((n, 4), np.float32); pred[0] = gt[init]
    times = np.empty(n - 1, np.float32)

    for i in range(1, n):
        t0 = time.perf_counter()
        pred[i] = track(tracker, frames[init + i], amp)['target_bbox']
        times[i-1] = time.perf_counter() - t0

    vgt, vex = gt[init:init+n], exist[init:init+n]
    ious = iou_batch(pred, vgt)
    vis = vex > 0; vc = int(vis.sum())

    if vc:
        auc = np.mean(np.mean(ious[vis,None] >= np.arange(0,1.05,0.05)[None,:], 0))*100
        sa_05 = np.mean(ious[vis]>=0.5)*100
        pc, gc = pred[vis,:2]+pred[vis,2:]/2, vgt[vis,:2]+vgt[vis,2:]/2
        prec = np.mean(np.sqrt(((pc-gc)**2).sum(1))<=20)*100
    else:
        auc = sa_05 = prec = 0.0

    areas = pred[:,2]*pred[:,3]
    sa = np.mean(np.where(vis, ious, np.where(areas>100,1,0)))*100
    fps = 1.0/np.mean(times) if n>1 else 0
    return {'sa':sa,'auc':auc,'precision':prec,'sa_05':sa_05,'fps':fps,'n':n,'vis':vc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='config name, e.g. sutrack_b224_antiuav_rti')
    p.add_argument('checkpoint', help='checkpoint filename, e.g. SUTRACK_ep0090.pth.tar')
    p.add_argument('--test-dir', default=f'{prj}/data/AntI-UAV/test')
    p.add_argument('--output-dir', default=f'{prj}/output')
    args = p.parse_args()

    model_path = os.path.join(args.output_dir, args.config.replace('sutrack_',''),
                              'checkpoints/train/sutrack', args.config, args.checkpoint)
    if not os.path.exists(model_path):
        print(f"Not found: {model_path}"); return

    print(f"{args.config} | {args.checkpoint}", flush=True)
    params = parameters(args.config); params.checkpoint = model_path; params.debug = False

    seqs = sorted([(d, os.path.join(args.test_dir, d)) for d in os.listdir(args.test_dir)
                   if os.path.isdir(os.path.join(args.test_dir, d))
                   and os.path.exists(os.path.join(args.test_dir, d, 'infrared.mp4'))])
    if not seqs:
        print(f"No sequences in {args.test_dir}"); return

    results = {}
    all_sa, all_auc, all_prec, all_sa05, all_fps = [], [], [], [], []
    print(f"Seqs: {len(seqs)}", flush=True)
    t0 = time.time()
    tracker = SUTRACK(params, 'GOT10K')

    for i, (name, d) in enumerate(seqs):
        with open(os.path.join(d, 'infrared.json')) as f:
            data = json.load(f)
        exist = np.array(data['exist'], np.float32)
        gt = np.array([r[:4] for r in data.get('gt_rect',[]) if isinstance(r,list) and len(r)>=4], np.float32)

        r = eval_seq(tracker, d, gt, exist, USE_AMP)
        if r:
            results[name] = r
            all_sa.append(r['sa']); all_auc.append(r['auc'])
            all_prec.append(r['precision']); all_sa05.append(r['sa_05']); all_fps.append(r['fps'])
        if (i+1)%10==0: print(f"  [{i+1}/{len(seqs)}]", flush=True)

    t = time.time()-t0
    if all_sa:
        sa=np.mean(all_sa); auc=np.mean(all_auc); prec=np.mean(all_prec)
        sa05=np.mean(all_sa05); fps=np.mean(all_fps)
        print(f"\n{'='*55}")
        print(f"SA:{sa:.2f}%  AUC:{auc:.2f}%  Prec:{prec:.2f}%  SA@0.5:{sa05:.2f}%  FPS:{fps:.1f}  Time:{t:.0f}s")
        print(f"{'='*55}")
        out = {'model':model_path,'config':args.config,'modality':'infrared','algorithm':'SA',
               'num_sequences':len(seqs),'total_time':t,
               'average':{'sa':float(sa),'auc':float(auc),'precision':float(prec),
                          'sa_05':float(sa05),'fps':float(fps)},'per_sequence':results}
        fname = f'eval_{args.config}_{args.checkpoint.replace(".pth.tar","").replace("SUTRACK_","")}.json'
        with open(fname,'w') as f: json.dump(out,f,indent=2)
        print(f"Saved: {fname}")

if __name__ == '__main__':
    main()
