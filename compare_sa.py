#!/usr/bin/env python3
"""Per-sequence SA comparison: Standard vs MOT-v2"""
import json

a = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_standard_sa_infrared_fast.json'))
b = json.load(open(r'E:\biyesheji\SUTrack-main11\evaluation_results_mot_v2_ep0160.json'))

seqs_a = a['per_sequence']
seqs_b = b['per_sequence']

print(f"{'='*70}")
print(f"Per-Sequence SA Comparison: Standard vs MOT-v2 (TTE+STCA)")
print(f"{'='*70}")
print(f"Standard avg: {a['average']['sa']:.2f}%  |  MOT-v2 avg: {b['average']['sa']:.2f}%  |  Δ: {b['average']['sa'] - a['average']['sa']:+.2f}%")
print(f"{'='*70}")
print(f"{'Sequence':<35} {'Standard SA':>10} {'MOT-v2 SA':>10} {'Δ':>8} {'Winner':>12}")
print(f"{'-'*75}")

better = 0
worse = 0

for name in sorted(seqs_a.keys()):
    sa_a = seqs_a[name]['sa']
    sa_b = seqs_b.get(name, {}).get('sa', 0)
    delta = sa_b - sa_a
    w = 'MOT-v2' if delta > 0 else ('Standard' if delta < 0 else 'TIE')
    if delta > 0: better += 1
    elif delta < 0: worse += 1
    print(f"{name:<35} {sa_a:10.2f}% {sa_b:10.2f}% {delta:+8.2f}% {w:>12}")

print(f"{'='*70}")
print(f"MOT-v2 wins: {better} seqs  |  Standard wins: {worse} seqs  |  Ties: {91 - better - worse}")
print(f"{'='*70}")

# Top 10 where MOT-v2 wins most
diffs = [(name, seqs_b[name]['sa'] - seqs_a[name]['sa']) for name in seqs_a if name in seqs_b]
diffs.sort(key=lambda x: x[1])
print(f"\n--- Top 10: MOT-v2 > Standard (biggest wins) ---")
for name, d in sorted(diffs, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name}: +{d:.2f}% (MOT:{seqs_b[name]['sa']:.1f}% vs Std:{seqs_a[name]['sa']:.1f}%)")

print(f"\n--- Bottom 10: Standard > MOT-v2 (biggest losses) ---")
for name, d in diffs[:10]:
    print(f"  {name}: {d:+.2f}% (MOT:{seqs_b[name]['sa']:.1f}% vs Std:{seqs_a[name]['sa']:.1f}%)")

# Distribution
import numpy as np
d_arr = np.array([d for _, d in diffs])
print(f"\n--- Δ Distribution ---")
print(f"  Mean: {d_arr.mean():+.2f}%")
print(f"  Std: {d_arr.std():.2f}%")
print(f"  Min: {d_arr.min():+.2f}%")
print(f"  Max: {d_arr.max():+.2f}%")
bins = [(-100, -20), (-20, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 100)]
for lo, hi in bins:
    cnt = np.sum((d_arr >= lo) & (d_arr < hi))
    print(f"  Δ in [{lo:+3},{hi:+3}): {cnt:2d} seqs")
