"""G3 on MATH (numeric subset), CPU llama.cpp: greedy CoT + 3 PAL samples (T=0.7), plurality vote, ties to greedy CoT.
Resumable: appends one json line per question to rows.jsonl. Parsing/data identical to ../m3-math/run.py."""
import argparse, json, os, random, sys, time
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3-math"))
import run as M  # boxed, num, extract, run_code, same, key, load_math, SYS_COT, SYS_PAL

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
ap.add_argument("--offset", type=int, default=2000); ap.add_argument("--n", type=int, default=250)
ap.add_argument("--k", type=int, default=3); ap.add_argument("--seed", type=int, default=8181)
ap.add_argument("--out", default="rows.jsonl"); ap.add_argument("--max_new", type=int, default=1024)
a = ap.parse_args()
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download(a.repo, a.file), n_ctx=2048, n_threads=4, verbose=False, seed=a.seed)
df = M.load_math("test")
idx = list(range(len(df))); random.Random(2029).shuffle(idx); idx = idx[a.offset:a.offset + a.n]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["idx"] for l in open(a.out) if l.strip()}
def chat(sys_, q, temp, seed):
    r = llm.create_chat_completion([{"role": "system", "content": sys_}, {"role": "user", "content": q}],
                                   temperature=temp, max_tokens=a.max_new, seed=seed)
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
for j, i in enumerate(idx):
    if i in done: continue
    t0 = time.time(); q = df.question[i]; g = float(df.gold[i])
    ct, cn = chat(M.SYS_COT, q, 0.0, a.seed)
    c = M.extract(ct)
    pals, ptok = [], 0
    for s in range(a.k):
        pt, pn = chat(M.SYS_PAL, q, 0.7, a.seed * 10 + s + 1000 * j); ptok += pn
        pals.append(M.run_code(pt))
    votes = Counter(M.key(x) for x in [c] + pals if x is not None)
    g3 = c
    if votes:
        top = max(votes.values()); best = [v for v, n in votes.items() if n == top]
        if M.key(c) not in best: g3 = best[0] if len(best) == 1 else (pals and next(x for x in pals if M.key(x) in best))
    row = dict(idx=i, subject=df.subject[i], level=df.level[i], gold=g, cot=c, cot_ok=M.same(c, g), boxed=M.boxed(ct) is not None,
               cot_tok=cn, pal=pals, pal_ok=[M.same(x, g) for x in pals], pal_tok=ptok, g3=g3, g3_ok=M.same(g3, g),
               cot_tail=ct[-300:], secs=round(time.time() - t0, 1))
    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    print(j, row["cot_ok"], row["pal_ok"], row["g3_ok"], row["secs"], flush=True)
