#!/usr/bin/env python3
"""Standard vs RTI per-sequence SA comparison"""
import json, numpy as np

std = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_standard_sa_infrared_fast.json'))
rti = json.load(open(r'E:\biyesheji\SUTrack-main11\eval_sutrack_b224_antiuav_rti_ep0180.json'))

s = std['per_sequence']
r = rti['per_sequence']

print(f"Standard: SA={std['average']['sa']:.2f}%  |  RTI: SA={rti['average']['sa']:.2f}%  |  \u0394: {rti['average']['sa']-std['average']['sa']:+.2f}%")
print(f"{'='*55}")

diffs = []
for name in sorted(s):
    sa_s = s[name]['sa']
    sa_r = r.get(name, {}).get('sa', 0)
    d = sa_r - sa_s
    diffs.append((name, d, sa_s, sa_r))
    w = 'RTI' if d > 0 else ('Std' if d < 0 else '=')
    print(f"{name:<30} {sa_s:6.1f}% {sa_r:6.1f}% {d:+7.2f}% {w}")

# Stats
d_arr = np.array([d for _, d, _, _ in diffs])
wins = np.sum(d_arr > 0)
loss = np.sum(d_arr < 0)
print(f"\nRTI wins: {wins}  |  Standard wins: {loss}  |  Mean \u0394: {d_arr.mean():+.2f}%")

# Top/Bottom
print(f"\n--- Top 10 RTI wins ---")
for name, d, sa_s, sa_r in sorted(diffs, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name}: +{d:.1f}% (RTI:{sa_r:.1f} vs Std:{sa_s:.1f})")
print(f"\n--- Bottom 10 Standard wins ---")
for name, d, sa_s, sa_r in sorted(diffs, key=lambda x: x[1])[:10]:
    print(f"  {name}: {d:+.1f}% (RTI:{sa_r:.1f} vs Std:{sa_s:.1f})")

# MOT-v2 disaster check
disaster = '20190925_193610'
d_s = [d for n, d, _, _ in diffs if n.startswith(disaster)]
if d_s:
    print(f"\n--- 193610 disaster seqs (MOT-v2 avg -47%) ---")
    print(f"  RTI avg \u0394 in 193610: {np.mean(d_s):+.1f}%")
    for name, d, sa_s, sa_r in diffs:
        if name.startswith(disaster):
            print(f"  {name}: {d:+.1f}% (RTI:{sa_r:.1f} vs Std:{sa_s:.1f})")
