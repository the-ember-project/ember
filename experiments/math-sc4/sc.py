"""Two extra sampled CoTs (CS2, CS3) per math-g3 question (1.5B, positions 2000-2249), T=0.7, fresh seed 9191. Resumable."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3-math"))
import run as M
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../math-g3/rows.jsonl"); ap.add_argument("--n", type=int, default=250)
ap.add_argument("--seed", type=int, default=9191); ap.add_argument("--out", default="rows.jsonl")
ap.add_argument("--max_new", type=int, default=1024)
a = ap.parse_args()
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=2048, n_threads=4, verbose=False, seed=a.seed)
df = M.load_math("test")
src = [json.loads(l) for l in open(a.src) if l.strip()][:a.n]
done = {json.loads(l)["idx"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
for j, r in enumerate(src):
    i = r["idx"]
    if i in done: continue
    t0 = time.time(); q = df.question[i]; g = float(df.gold[i]); assert abs(g - r["gold"]) < 1e-9
    row = dict(idx=i, gold=g)
    for s, off in (("cs2", 0), ("cs3", 5000)):
        x = llm.create_chat_completion([{"role": "system", "content": M.SYS_COT}, {"role": "user", "content": q}],
                                       temperature=0.7, max_tokens=a.max_new, seed=a.seed * 10 + off + j)
        t = x["choices"][0]["message"]["content"]; v = M.extract(t)
        row.update({s: v, s + "_ok": M.same(v, g), s + "_boxed": M.boxed(t) is not None,
                    s + "_tok": x["usage"]["completion_tokens"], s + "_tail": t[-300:]})
    row["secs"] = round(time.time() - t0, 1)
    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    print(j, row["cs2_ok"], row["cs3_ok"], row["secs"], flush=True)
