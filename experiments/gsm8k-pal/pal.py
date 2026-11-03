"""Program-aided GSM8K (PAL / program-of-thought): model writes Python, we execute it.
Same question sampling as ../gsm8k-baseline/eval.py (seed 1234 shuffle, --offset/--n). Greedy, zero-shot.
Records the program, its stdout/error and the extracted answer; resumable."""
import argparse, json, os, random, re, subprocess, sys, tempfile, time
from huggingface_hub import hf_hub_download
import pandas as pd
from llama_cpp import Llama

SYSTEM = ("You are a careful math solver. Write a short Python program that computes the answer to the "
          "problem. Use clear variable names and comments for each step. Store the final numeric answer "
          "in a variable named `answer` and end with `print(answer)`. Output only the code in one "
          "```python block.")

def run_code(text, timeout=10):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[0] if m else text
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        out, err = p.stdout.strip(), p.stderr.strip()[-300:]
    except subprocess.TimeoutExpired:
        out, err = "", "TIMEOUT"
    os.unlink(path)
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", out)
    try:
        pred = float(nums[-1].replace(",", "")) if nums else None
    except ValueError:
        pred = None
    return code, out[-200:], err, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mp = hf_hub_download(a.repo, a.file)
    df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    idx = list(range(len(df))); random.Random(a.seed).shuffle(idx); idx = idx[a.offset:a.offset + a.n]
    llm = Llama(model_path=mp, n_ctx=2048, n_threads=os.cpu_count(), verbose=False, seed=0)
    done = {json.loads(l)["idx"] for l in open(a.out)} if os.path.exists(a.out) else set()
    c = t_ = 0
    with open(a.out, "a") as f:
        for i in idx:
            if i in done: continue
            gold = float(df.answer[i].split("####")[-1].strip().replace(",", ""))
            t = time.time()
            r = llm.create_chat_completion(messages=[{"role": "system", "content": SYSTEM},
                                                     {"role": "user", "content": df.question[i]}],
                                           temperature=0.0, max_tokens=a.max_tokens)
            text = r["choices"][0]["message"]["content"]
            code, out, err, pred = run_code(text)
            ok = pred is not None and abs(pred - gold) < 1e-4
            c += ok; t_ += 1
            f.write(json.dumps({"idx": i, "gold": gold, "pred": pred, "ok": ok, "secs": round(time.time() - t, 2),
                                "tokens": r["usage"]["completion_tokens"], "exec_ok": not err, "err": err,
                                "stdout": out, "output": text}) + "\n"); f.flush()
            print(f"{t_}: idx={i} ok={ok} pred={pred} gold={gold} err={bool(err)} acc={c/t_:.3f}", flush=True)
    print(f"SUMMARY n={t_} acc={c/max(t_,1):.3f}")

if __name__ == "__main__":
    main()
