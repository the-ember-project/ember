"""math-mix7: greedy CoT + 3 PAL (T=0.7) + 3 sampled CoT (T=0.7) per MATH numeric test question, fresh offsets
(2029 shuffle, positions 2450+; 2000-2449 were used by math-g3/-3b/-xscale/-sc4). 1.5B Q4. Resumable."""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3-math"))
import run as M
ap = argparse.ArgumentParser()
ap.add_argument("--offset", type=int, default=2450); ap.add_argument("--n", type=int, default=250)
ap.add_argument("--seed", type=int, default=7171); ap.add_argument("--out", default="rows.jsonl")
ap.add_argument("--max_new", type=int, default=1024)
a = ap.parse_args()
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=2048, n_threads=4, verbose=False, seed=a.seed)
df = M.load_math("test")
idx = list(range(len(df))); random.Random(2029).shuffle(idx); idx = idx[a.offset:a.offset + a.n]
done = {json.loads(l)["idx"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
def chat(sys_, q, temp, seed):
    r = llm.create_chat_completion([{"role": "system", "content": sys_}, {"role": "user", "content": q}],
                                   temperature=temp, max_tokens=a.max_new, seed=seed)
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
for j, i in enumerate(idx):
    if i in done: continue
    t0 = time.time(); q = df.question[i]; g = float(df.gold[i])
    ct, cn = chat(M.SYS_COT, q, 0.0, a.seed)
    row = dict(idx=i, subject=df.subject[i], level=df.level[i], gold=g, cot=M.extract(ct), boxed=M.boxed(ct) is not None,
               cot_tok=cn, cot_tail=ct[-300:], pal=[], pal_tok=[], pal_tail=[], cs=[], cs_tok=[], cs_boxed=[], cs_tail=[])
    for s in range(3):
        pt, pn = chat(M.SYS_PAL, q, 0.7, a.seed * 10 + s + 1000 * (a.offset + j))
        try: v = M.run_code(pt)
        except Exception: v = None
        row["pal"].append(v); row["pal_tok"].append(pn); row["pal_tail"].append(pt[-300:])
    for s in range(3):
        t, n = chat(M.SYS_COT, q, 0.7, a.seed * 10 + 500 + s + 1000 * (a.offset + j))
        row["cs"].append(M.extract(t)); row["cs_tok"].append(n); row["cs_boxed"].append(M.boxed(t) is not None)
        row["cs_tail"].append(t[-300:])
    row["secs"] = round(time.time() - t0, 1)
    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    print(j, M.same(row["cot"], g), [M.same(x, g) for x in row["pal"]], [M.same(x, g) for x in row["cs"]], row["secs"], flush=True)
