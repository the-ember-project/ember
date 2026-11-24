# Score each GSM8K solution (1.5B greedy from g3confirm + 7 samples) with a free reward model on CPU. Resumable.
import argparse, json, math, os, time
import pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--rm", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
ap.add_argument("--greedy", default="../gsm8k-g3confirm/cot_greedy_off900.jsonl")
ap.add_argument("--samples", default="cot_k7.jsonl")
ap.add_argument("--out", default="rm17_scores.jsonl")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
torch.set_num_threads(os.cpu_count())
df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
G = {r["idx"]: r for r in map(json.loads, open(a.greedy))}
S = [json.loads(l) for l in open(a.samples)]
done = {json.loads(l)["idx"] for l in open(a.out)} if os.path.exists(a.out) else set()
tok = AutoTokenizer.from_pretrained(a.rm)
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForSequenceClassification.from_pretrained(a.rm, dtype=torch.float32, num_labels=1).eval()
n = 0
with open(a.out, "a") as f:
    for r in S:
        i = r["idx"]
        if i in done or i not in G: continue
        texts = [G[i]["output"]] + list(r["output"])
        sc = []
        for t in texts:  # batch of 1: no padding effects
            s = tok.apply_chat_template([{"role": "user", "content": df.question[i]}, {"role": "assistant", "content": t}], tokenize=False)
            enc = tok(s, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
            t0 = time.time()
            with torch.no_grad(): v = m(**enc).logits[0, 0].float().item()
            sc.append(v if math.isfinite(v) else None)
        f.write(json.dumps({"idx": i, "scores": sc}) + "\n"); f.flush()
        n += 1; print(f"{n}: idx={i} {['%.2f' % x if x is not None else 'nan' for x in sc]}", flush=True)
        if a.limit and n >= a.limit: break
