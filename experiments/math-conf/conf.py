"""math-conf: token-level confidence of Qwen2.5-1.5B-Instruct (Q4, llama.cpp, CPU) on its own MATH samples from
math-verify (greedy + 7 at T=0.7, vLLM fp16). One prefill per sample with full logits; per sample we store DeepConf-style
confidences (token conf = -mean top-20 logprob; group = sliding window of 64 tokens) and self-certainty
(mean KL(U||p)). Only 'mixed' questions (>=2 distinct non-None answers) are scored: selection cannot change the rest.
Resumable: one jsonl row per question."""
import argparse, gzip, json, math, os, time
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--gen", default=os.path.join(H, "..", "math-verify", "gpu-output", "out", "gen_1.5B.jsonl.gz"))
ap.add_argument("--out", default=os.path.join(H, "conf_1.5B.jsonl")); ap.add_argument("--n", type=int, default=10**9)
a = ap.parse_args()
SYS_COT = ("You are a careful math solver. Reason step by step, then give the final answer "
           "(a number: integer, decimal or fraction) in the form \\boxed{answer}.")
def key(v): return None if v is None or not isinstance(v, (int, float)) or not math.isfinite(v) else round(v, 4)
def mixed(r): return len({key(x) for x in r["ans"]} - {None}) >= 2
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=3200, n_batch=512, n_threads=4, logits_all=True, verbose=False)
V = llm.n_vocab(); W = 64
def feats(problem, text):
    p = (f"<|im_start|>system\n{SYS_COT}<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n")
    pt = llm.tokenize(p.encode(), add_bos=False, special=True)
    ct = llm.tokenize(text.encode(), add_bos=False, special=False)[: 3200 - len(pt) - 1]
    if not ct: return None
    llm.reset(); llm.eval(pt + ct)
    lg = llm.scores[len(pt) - 1: len(pt) + len(ct) - 1]  # filled per batch when logits_all=True
    conf, cert, lp = [], [], []
    for s in range(0, len(ct), 256):  # chunked log-softmax to bound memory
        x = np.array(lg[s:s + 256], dtype=np.float32); x -= x.max(1, keepdims=True)
        x -= np.log(np.exp(x).sum(1, keepdims=True))
        top = -np.partition(-x, 20, axis=1)[:, :20]
        conf += list(-top.mean(1)); cert += list(-math.log(V) - x.mean(1))
        lp += list(x[np.arange(len(x)), ct[s:s + 256]])
    conf = np.array(conf, dtype=np.float64)
    if len(conf) >= W: grp = np.convolve(conf, np.ones(W) / W, mode="valid")
    else: grp = np.array([conf.mean()])
    k = max(1, int(math.ceil(0.1 * len(grp))))
    out = dict(n=len(ct), mean_conf=float(conf.mean()), bottom10=float(np.sort(grp)[:k].mean()), lowest=float(grp.min()),
               tail=float(conf[-256:].mean()), self_cert=float(np.mean(cert)), mean_lp=float(np.mean(lp)))
    return out if all(math.isfinite(v) for v in out.values()) else None
rows = [json.loads(l) for l in gzip.open(a.gen, "rt")]
todo = [r for r in rows if mixed(r)][: a.n]
print(f"{len(rows)} items, {len(todo)} mixed", flush=True)
done = set()
if os.path.exists(a.out):  # drop a partial last line left by a killed run
    good = []
    for l in open(a.out):
        try: good.append(json.loads(l))
        except Exception: pass
    open(a.out, "w").write("".join(json.dumps(r) + "\n" for r in good)); done = {r["pos"] for r in good}
with open(a.out, "a") as f:
    for r in todo:
        if r["pos"] in done: continue
        t0 = time.time(); fs = []
        for t in r["texts"]:
            try: fs.append(feats(r["problem"], t))
            except Exception as e: print("ERR", r["pos"], repr(e)[:200], flush=True); fs.append(None)
        f.write(json.dumps(dict(pos=r["pos"], idx=r["idx"], feats=fs, secs=round(time.time() - t0, 1))) + "\n"); f.flush()
        print(r["pos"], round(time.time() - t0, 1), flush=True)
