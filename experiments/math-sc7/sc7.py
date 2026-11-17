"""math-sc7: 3 MORE sampled CoT (T=0.7, new seeds) on math-mix7's 250 MATH items, so SC7 (greedy + 6 sampled CoT)
can be compared with MIX7 (greedy + 3 sampled CoT + 3 PAL) at equal voter count. 1.5B Q4. Resumable."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m3-math"))
import run as M
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=250); ap.add_argument("--seed", type=int, default=9393)
ap.add_argument("--out", default="cs2.jsonl"); ap.add_argument("--max_new", type=int, default=1024)
a = ap.parse_args()
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=2048, n_threads=4, verbose=False, seed=a.seed)
df = M.load_math("test")
base = [json.loads(l) for l in open(os.path.join(HERE, "..", "math-mix7", "rows.jsonl")) if l.strip()][:a.n]
done = {json.loads(l)["idx"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
for j, b in enumerate(base):
    i = b["idx"]
    if i in done: continue
    t0 = time.time(); q = df.question[i]; g = float(df.gold[i]); assert abs(g - b["gold"]) < 1e-9
    row = dict(idx=i, cs2=[], cs2_tok=[], cs2_boxed=[], cs2_tail=[])
    for s in range(3):
        r = llm.create_chat_completion([{"role": "system", "content": M.SYS_COT}, {"role": "user", "content": q}],
                                       temperature=0.7, max_tokens=a.max_new, seed=a.seed * 10 + s + 1000 * j)
        t, n = r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
        row["cs2"].append(M.extract(t)); row["cs2_tok"].append(n); row["cs2_boxed"].append(M.boxed(t) is not None)
        row["cs2_tail"].append(t[-300:])
    row["secs"] = round(time.time() - t0, 1)
    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    print(j, [M.same(x, g) for x in row["cs2"]], row["secs"], flush=True)
