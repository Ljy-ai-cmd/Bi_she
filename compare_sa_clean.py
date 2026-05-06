#!/usr/bin/env python3
"""对比：排除 193610 灾难序列后的 SA"""
import json, numpy as np

a = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_standard_sa_infrared_fast.json'))
b = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_mot_v2_ep0160.json'))

seqs_a = a['per_sequence']
seqs_b = b['per_sequence']

# 193610 灾难序列
disaster_prefix = '20190925_193610'
disaster_seqs = [n for n in seqs_a if n.startswith(disaster_prefix)]

# 排除灾难序列
sa_a_all = [v['sa'] for v in seqs_a.values()]
sa_a_clean = [v['sa'] for n, v in seqs_a.items() if not n.startswith(disaster_prefix)]

sa_b_all = [seqs_b[n]['sa'] for n in seqs_a if n in seqs_b]
sa_b_clean = [seqs_b[n]['sa'] for n in seqs_a if n in seqs_b and not n.startswith(disaster_prefix)]

print(f"{'='*60}")
print(f"Impact of Removing 193610 Disaster Sequences")
print(f"{'='*60}")
print(f"Disaster sequences: {disaster_seqs}")
print()

print(f"{'Metric':<35} {'Standard':>10} {'MOT-v2':>10} {'Δ':>8}")
print(f"{'-'*65}")
print(f"{'All 91 seqs avg SA':<35} {np.mean(sa_a_all):10.2f}% {np.mean(sa_b_all):10.2f}% {np.mean(sa_b_all)-np.mean(sa_a_all):+8.2f}%")
print(f"{'After removing 193610 (82 seqs)':<35} {np.mean(sa_a_clean):10.2f}% {np.mean(sa_b_clean):10.2f}% {np.mean(sa_b_clean)-np.mean(sa_a_clean):+8.2f}%")
print(f"{'-'*65}")

# 193610 序列单独看
print(f"\n{'='*60}")
print(f"193610 Sequences Detail")
print(f"{'='*60}")
print(f"{'Sequence':<30} {'Standard':>10} {'MOT-v2':>10} {'Δ':>8}")
for n in sorted(disaster_seqs):
    sa_a = seqs_a[n]['sa']
    sa_b = seqs_b.get(n, {}).get('sa', 0)
    print(f"{n:<30} {sa_a:10.2f}% {sa_b:10.2f}% {sa_b-sa_a:+8.2f}%")
print(f"{'-'*45}")
print(f"{'Average':<30} {np.mean([seqs_a[n]['sa'] for n in disaster_seqs]):10.2f}% "
      f"{np.mean([seqs_b[n]['sa'] for n in disaster_seqs]):10.2f}% "
      f"{np.mean([seqs_b[n]['sa']-seqs_a[n]['sa'] for n in disaster_seqs]):+8.2f}%")

# 极端坏序列的 MOT-v2 值
print(f"\n--- MOT-v2 worst SA in 193610 ---")
worst = sorted([(n, seqs_b[n]['sa']) for n in disaster_seqs], key=lambda x: x[1])
for n, s in worst:
    print(f"  {n}: {s:.1f}%")
