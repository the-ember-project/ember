"""GSM8K eval for a GGUF model with llama-cpp-python.
Usage: python eval.py --n 100 --out results_qwen1.5b.jsonl [--repo ... --file ...] [--offset 0]
"""
import argparse, json, os, random, re, time
from collections import Counter
_h = os.environ.get("HF_HOME", "")
if not _h or not os.access(_h, os.W_OK):  # session sandbox can't write the shared cache
    os.environ["HF_HOME"] = "/tmp/hf"
from huggingface_hub import hf_hub_download
import pandas as pd
from llama_cpp import Llama

SYSTEM = ("You are a careful math solver. Reason step by step, then give the final "
          "answer on its own line in the form '#### <number>'.")

def extract(text):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if m:
        s = m[-1]
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "")
    try:
        v = float(s)
        return v
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1, help="samples per question (majority vote if >1)")
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    mp = hf_hub_download(a.repo, a.file)
    dp = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(dp)
    idx = list(range(len(df)))
    random.Random(a.seed).shuffle(idx)
    idx = idx[a.offset:a.offset + a.n]

    llm = Llama(model_path=mp, n_ctx=2048, n_threads=a.threads, verbose=False, seed=0)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            done.add(json.loads(line)["idx"])
    correct = total = 0; toks = 0; gen_time = 0.0
    with open(a.out, "a") as f:
        for i in idx:
            if i in done:
                continue
            q = df.question[i]; gold = float(df.answer[i].split("####")[-1].strip().replace(",", ""))
            t = time.time()
            texts, ntok = [], 0
            for j in range(a.k):
                r = llm.create_chat_completion(messages=[{"role": "system", "content": SYSTEM},
                                                         {"role": "user", "content": q}],
                                               temperature=a.temp, max_tokens=a.max_tokens,
                                               seed=a.seed * 1000 + j)
                texts.append(r["choices"][0]["message"]["content"])
                ntok += r["usage"]["completion_tokens"]
            dt = time.time() - t
            preds = [extract(x) for x in texts]
            votes = Counter(p for p in preds if p is not None)
            pred = votes.most_common(1)[0][0] if votes else None  # ties -> earliest sample
            ok = pred is not None and abs(pred - gold) < 1e-6
            correct += ok; total += 1; toks += ntok; gen_time += dt
            f.write(json.dumps({"idx": i, "gold": gold, "pred": pred, "ok": ok, "secs": round(dt, 2),
                                "tokens": ntok, "preds": preds,
                                "output": texts[0] if a.k == 1 else texts}) + "\n"); f.flush()
            print(f"{total}: idx={i} ok={ok} pred={pred} gold={gold} {dt:.1f}s {ntok}tok  acc={correct/total:.3f}", flush=True)
    if total:
        print(f"SUMMARY n={total} acc={correct/total:.3f} tok/s={toks/gen_time:.1f} s/q={gen_time/total:.1f}")

if __name__ == "__main__":
    main()
