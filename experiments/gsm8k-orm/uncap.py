# Continue outputs that hit the 400-token cap up to 1024 total tokens (equivalent to a 1024 cap). Resumable.
# Usage: python uncap.py --src cot3b_greedy.jsonl --dst cot3b_greedy_u.jsonl --repo ... --file ... --temp 0
import argparse, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gsm8k-baseline"))
from eval import SYSTEM, extract
from huggingface_hub import hf_hub_download
import pandas as pd
from llama_cpp import Llama

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=7171)
ap.add_argument("--thr", type=int, default=395)
ap.add_argument("--cap", type=int, default=1024)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--selftest", action="store_true", help="truncate a finished greedy output and check continuation")
a = ap.parse_args()
df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
llm = Llama(model_path=hf_hub_download(a.repo, a.file), n_ctx=2048, n_threads=os.cpu_count(), verbose=False, seed=0)
ntok = lambda t: len(llm.tokenize(t.encode(), add_bos=False, special=False))
pre = lambda q: f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

def cont(q, text, seed):
    r = llm.create_completion(prompt=pre(q) + text, max_tokens=max(1, a.cap - ntok(text)), temperature=a.temp,
                              seed=seed, stop=["<|im_end|>", "<|endoftext|>"])
    return r["choices"][0]["text"], r["usage"]["completion_tokens"]

src = [json.loads(l) for l in open(a.src)][:a.limit]
if a.selftest:
    r = next(x for x in src if 150 < ntok(x["output"] if isinstance(x["output"], str) else x["output"][0]) < 350)
    t = r["output"] if isinstance(r["output"], str) else r["output"][0]
    toks = llm.tokenize(t.encode(), add_bos=False, special=False); h = llm.detokenize(toks[:len(toks) // 2]).decode()
    c, _ = cont(df.question[r["idx"]], h, 0)
    print("SELFTEST", "MATCH" if (h + c).strip() == t.strip() else "DIFF", repr(t[-80:]), "|", repr((h + c)[-80:]))
    sys.exit(0)
done = {json.loads(l)["idx"] for l in open(a.dst)} if os.path.exists(a.dst) else set()
n = nc = 0
with open(a.dst, "a") as f:
    for r in src:
        if r["idx"] in done: continue
        q = df.question[r["idx"]]
        outs = [r["output"]] if isinstance(r["output"], str) else list(r["output"])
        added, still = [], 0
        for j, t in enumerate(outs):
            if ntok(t) >= a.thr:
                c, k = cont(q, t, a.seed * 1000 + j); outs[j] = t + c; added.append(k); nc += 1
                still += ntok(outs[j]) >= a.cap - 5
            else: added.append(0)
        preds = [extract(t) for t in outs]
        votes = Counter(p for p in preds if p is not None)
        pred = votes.most_common(1)[0][0] if votes else None
        r2 = dict(r, preds=preds, pred=pred, ok=pred is not None and abs(pred - r["gold"]) < 1e-6,
                  tokens=r["tokens"] + sum(added), added=added, still_capped=still,
                  output=outs[0] if isinstance(r["output"], str) else outs)
        f.write(json.dumps(r2) + "\n"); f.flush(); n += 1
        if any(added): print(f"{n}: idx={r['idx']} continued {sum(x > 0 for x in added)} ok {r['ok']}->{r2['ok']} still {still}", flush=True)
print(f"UNCAP {a.src}: {n} items, {nc} outputs continued")
