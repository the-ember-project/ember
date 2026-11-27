"""Stratified analysis of MIXR-DPO test gains by BASE's own sample answers (see README)."""
import json, gzip, os, collections
H = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(H, '..')
def rd(p):
    op = gzip.open if p.endswith('.gz') else open
    with op(p, 'rt') as f:
        return {r['idx']: r for r in map(json.loads, f)}
def eq(a, b):
    try: return a is not None and b is not None and abs(float(a) - float(b)) < 1e-6
    except Exception: return False
g3 = rd(f'{E}/m3-stack/gpu-output/test_g3_BASE.jsonl.gz')
seeds = {'R1': (f'{E}/m3-stack/gpu-output/test_eval_BASE.jsonl', f'{E}/m3-stack/gpu-output/test_eval_MIXR.jsonl')}
for s in ('R2', 'R3'):
    d = f'{E}/m3-dpo-rep/gpu-output/{s}'
    seeds[s] = (f'{d}/test_eval_BASE.jsonl', f'{d}/test_eval_MIXR.jsonl')
def stratum(i, base_correct):
    r = g3[i]; gold = r['gold']
    if base_correct: return 'AG' if r['correct'] else 'WG'
    if r['correct']: return 'A'
    if any(eq(p, gold) for p in r['pal']) or eq(r['greedy'], gold): return 'B'
    return 'C'
cnt = collections.Counter(); chg = collections.Counter(); p4 = [0, 0]; out = []
mism = {}
for s, (bp, mp) in seeds.items():
    B, M = rd(bp), rd(mp)
    mism[s] = sum(not eq(B[i]['pred'], g3[i]['greedy']) for i in B)
    for i in B:
        st = stratum(i, B[i]['correct'])
        cnt[st] += 1
        if B[i]['correct'] != M[i]['correct']:
            chg[st] += 1
            if st == 'WG':
                p4[1] += 1; p4[0] += eq(M[i]['pred'], g3[i]['g3'])
rate = {k: chg[k] / cnt[k] for k in cnt}
fixes = sum(chg[k] for k in 'ABC')
lines = [f'greedy mismatches vs m3-stack g3 greedy (BASE nondeterminism): {mism}',
         f'counts (item x seed): {dict(cnt)}', f'changes: {dict(chg)}',
         'rates: ' + ', '.join(f'{k} {100*rate[k]:.1f}%' for k in ['A', 'B', 'C', 'AG', 'WG'])]
lines.append(f"P1 fix A/C = {rate['A']/max(rate['C'],1e-9):.2f} (>=2) -> {'PASS' if rate['A'] >= 2*rate['C'] else 'FAIL'}")
lines.append(f"P2 break WG/AG = {rate['WG']/max(rate['AG'],1e-9):.2f} (>=2) -> {'PASS' if rate['WG'] >= 2*rate['AG'] else 'FAIL'}")
lines.append(f"P3 share of fixes from C = {100*chg['C']/fixes:.1f}% (<=30) -> {'PASS' if chg['C'] <= .3*fixes else 'FAIL'}")
lines.append(f"P4 WG breaks landing on BASE vote answer = {p4[0]}/{p4[1]} = {100*p4[0]/max(p4[1],1):.1f}% (>=50) -> {'PASS' if p4[0] >= .5*p4[1] else 'FAIL'}")
net = {k: chg[k] * (1 if k in 'ABC' else -1) / 3 for k in chg}
lines.append('net gain per seed by stratum (items): ' + ', '.join(f'{k} {net[k]:+.1f}' for k in ['A', 'B', 'C', 'AG', 'WG']))
txt = '\n'.join(lines); print(txt)
open(f'{H}/analysis.txt', 'w').write(txt + '\n')
