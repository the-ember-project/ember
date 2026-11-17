"""math-3b-sc: Qwen2.5-3B Q4 on math-mix7's MATH items (2029 shuffle, positions 2450-2699, n=250), same prompt
(m3-math SYS_COT) and limits as math-mix7. Phase 1: greedy CoT for all items (g.jsonl). Phase 2: 2 sampled CoT
(T=0.7) per item (s.jsonl). Resumable."""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3-math"))
import run as M
ap = argparse.ArgumentParser()
ap.add_argument("--offset", type=int, default=2450); ap.add_argument("--n", type=int, default=250)
ap.add_argument("--seed", type=int, default=6262); ap.add_argument("--max_new", type=int, default=1024)
a = ap.parse_args()
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
            n_ctx=2048, n_threads=4, verbose=False, seed=a.seed)
df = M.load_math("test")
idx = list(range(len(df))); random.Random(2029).shuffle(idx); idx = idx[a.offset:a.offset + a.n]
def done(f): return {json.loads(l)["idx"] for l in open(f) if l.strip()} if os.path.exists(f) else set()
def chat(q, temp, seed):
    r = llm.create_chat_completion([{"role": "system", "content": M.SYS_COT}, {"role": "user", "content": q}],
                                   temperature=temp, max_tokens=a.max_new, seed=seed)
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
for phase, f in (("g", "g.jsonl"), ("s", "s.jsonl")):
    d = done(f)
    for j, i in enumerate(idx):
        if i in d: continue
        t0 = time.time(); q = df.question[i]; g = float(df.gold[i]); row = dict(idx=i, gold=g, level=df.level[i])
        if phase == "g":
            t, n = chat(q, 0.0, a.seed)
            row.update(cot=M.extract(t), boxed=M.boxed(t) is not None, tok=n, tail=t[-300:])
        else:
            row.update(cs=[], tok=[], boxed=[], tail=[])
            for s in range(2):
                t, n = chat(q, 0.7, a.seed * 10 + s + 1000 * (a.offset + j))
                row["cs"].append(M.extract(t)); row["tok"].append(n); row["boxed"].append(M.boxed(t) is not None)
                row["tail"].append(t[-300:])
        row["secs"] = round(time.time() - t0, 1)
        with open(f, "a") as fh: fh.write(json.dumps(row) + "\n")
        print(phase, j, row["secs"], flush=True)
