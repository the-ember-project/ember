"""math-selfverify: Qwen2.5-1.5B-Instruct (Q4, llama.cpp) grades its own MATH solutions: score = P(Yes)/(P(Yes)+P(No))
for the first answer token. Mode 'tails': stored 300-char tails of the 7 CoT samples (math-mix7 rows + math-sc7 cs2).
Resumable: one jsonl row per question with 7 scores."""
import argparse, json, math, os, sys, time
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..")
sys.path.insert(0, os.path.join(R, "m3-math"))
import run as M
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="sv_tails.jsonl"); ap.add_argument("--n", type=int, default=250)
a = ap.parse_args()
import numpy as np
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=4096, n_threads=4, verbose=False)
SYS = "You are a strict math teacher who checks students' solutions."
def prompt(q, sol):
    u = (f"Problem:\n{q}\n\nStudent's solution (end part):\n...{sol}\n\n"
         "Is the student's final answer correct? Answer with a single word: Yes or No.")
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n")
YES = [llm.tokenize(w.encode(), add_bos=False, special=False)[0] for w in ("Yes", " Yes", "yes")]
NO = [llm.tokenize(w.encode(), add_bos=False, special=False)[0] for w in ("No", " No", "no")]
def score(q, sol):
    toks = llm.tokenize(prompt(q, sol).encode(), add_bos=False, special=True)
    llm.reset(); llm.eval(toks)
    lg = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64); lg -= lg.max(); p = np.exp(lg); p /= p.sum()
    y, n = float(p[YES].sum()), float(p[NO].sum())
    return (y / (y + n) if y + n > 0 else 0.5), y + n
B = {r["idx"]: r for r in (json.loads(l) for l in open(f"{R}/math-mix7/rows.jsonl") if l.strip())}
C = [json.loads(l) for l in open(f"{R}/math-sc7/cs2.jsonl") if l.strip()][:a.n]
df = M.load_math("test")
done = {json.loads(l)["idx"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
with open(a.out, "a") as f:
    for c in C:
        if c["idx"] in done: continue
        r = B[c["idx"]]; t0 = time.time(); q = df.question[c["idx"]]
        tails = [r["cot_tail"]] + r["cs_tail"] + c["cs2_tail"]
        res = [score(q, t) for t in tails]
        f.write(json.dumps(dict(idx=c["idx"], sv=[s for s, _ in res], mass=[m for _, m in res], secs=round(time.time() - t0, 1))) + "\n"); f.flush()
