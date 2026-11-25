# Re-score with the RM only the 1.5B solutions changed by uncap.py; copy the rest from rm17_scores.jsonl. Resumable.
import json, math, os, time
import pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RM = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
torch.set_num_threads(os.cpu_count())
df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
L = lambda p: {r["idx"]: r for r in map(json.loads, open(p))}
G0, S0 = L("../gsm8k-g3confirm/cot_greedy_off900.jsonl"), L("cot_k7.jsonl")
G1, S1, R0 = L("greedy15_u.jsonl"), L("cot_k7_u.jsonl"), L("rm17_scores.jsonl")
out = "rm17_scores_u.jsonl"
done = set(L(out)) if os.path.exists(out) else set()
tok = AutoTokenizer.from_pretrained(RM)
m = AutoModelForSequenceClassification.from_pretrained(RM, dtype=torch.float32, num_labels=1).eval()
nr = 0
with open(out, "a") as f:
    for i in S1:
        if i in done or i not in G1 or i not in R0: continue
        old = [G0[i]["output"]] + list(S0[i]["output"]); new = [G1[i]["output"]] + list(S1[i]["output"])
        sc = list(R0[i]["scores"])
        for j, (o, t) in enumerate(zip(old, new)):
            if o == t: continue
            s = tok.apply_chat_template([{"role": "user", "content": df.question[i]}, {"role": "assistant", "content": t}], tokenize=False)
            enc = tok(s, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
            with torch.no_grad(): v = m(**enc).logits[0, 0].float().item()
            sc[j] = v if math.isfinite(v) else None; nr += 1
        f.write(json.dumps({"idx": i, "scores": sc}) + "\n"); f.flush()
print(f"RESCORE: {nr} solutions re-scored")
