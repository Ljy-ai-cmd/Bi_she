#!/usr/bin/env python3
"""统计82序列完整指标 — Standard vs MOT-v2 vs CLAHE"""
import json, numpy as np

std = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_standard_sa_infrared_fast.json'))
mot = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_mot_v2_ep0160.json'))
clahe = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_clahe_sa_infrared.json'))

exclude = '20190925_193610'
s82 = {n: v for n, v in std['per_sequence'].items() if not n.startswith(exclude)}
m82 = {n: mot['per_sequence'][n] for n in s82 if n in mot['per_sequence']}
c82 = {n: clahe['per_sequence'][n] for n in s82 if n in clahe['per_sequence']}

keys = ['sa', 'auc', 'precision', 'sa_05', 'fps']
print("=== 82序列完整指标 (排除193610) ===")
print(f"{'指标':<15} {'Standard':>12} {'MOT-v2':>12} {'CLAHE':>12} {'Δ MOT':>10} {'Δ CLAHE':>10}")
print("-" * 75)
for k in keys:
    sv = np.mean([s82[n][k] for n in s82])
    mv = np.mean([m82[n][k] for n in m82])
    cv = np.mean([c82[n][k] for n in c82])
    print(f"{k:<15} {sv:12.2f} {mv:12.2f} {cv:12.2f} {mv-sv:+10.2f} {cv-sv:+10.2f}")

# SA Delta 分布 MOT vs Standard
ds_mot = [m82[n]['sa'] - s82[n]['sa'] for n in m82]
w_mot = sum(1 for d in ds_mot if d > 0); l_mot = sum(1 for d in ds_mot if d < 0)
ds_clahe = [c82[n]['sa'] - s82[n]['sa'] for n in c82]
w_clahe = sum(1 for d in ds_clahe if d > 0); l_clahe = sum(1 for d in ds_clahe if d < 0)

print(f"\n--- SA Delta 胜负统计 ---")
print(f"MOT-v2 wins: {w_mot}/{len(m82)}, losses: {l_mot}/{len(m82)}")
print(f"CLAHE  wins: {w_clahe}/{len(c82)}, losses: {l_clahe}/{len(c82)}")

print(f"\n--- SA Delta 分布 ---")
print(f"{'区间':<15} {'MOT-v2':>8} {'CLAHE':>8}")
for lo, hi in [(-20,-10),(-10,0),(0,10),(10,50)]:
    print(f"[{lo:+3}, {hi:+3}){'':<7} {sum(1 for d in ds_mot if lo<=d<hi):8} {sum(1 for d in ds_clahe if lo<=d<hi):8}")

# Top 5 MOT wins & losses
print(f"\n--- Top 5 MOT vs Standard (82 seqs) ---")
print(f"{'序列':<30} {'Standard':>8} {'MOT-v2':>8} {'Delta':>8}")
idx = np.argsort(ds_mot)[::-1][:5]
for i in idx:
    n = list(m82.keys())[i]
    print(f"{n:<30} {s82[n]['sa']:8.2f} {m82[n]['sa']:8.2f} {ds_mot[i]:+8.2f}")
