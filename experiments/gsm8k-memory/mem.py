"""Memory-of-Thought at 1.5B with a cross-format filter (GSM8K).
Phase 1 (bank): greedy CoT + greedy PAL on n_bank GSM8K *train* questions; keep CoT traces whose answer
  agrees with the PAL answer (label-free). Gold correctness is recorded only to report bank precision.
Phase 2 (eval): GSM8K test (seed-1234 shuffle, --offset/--n), greedy CoT in 3 arms per question:
  Z = zero-shot; R = 2 TF-IDF-nearest bank demos (as chat turns); X = 2 random bank demos.
Both phases resumable. Phase 2 starts only when the bank is complete."""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsm8k-baseline"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsm8k-pal"))
from eval import SYSTEM, extract  # noqa: E402  (also sets HF_HOME fallback)
from pal import SYSTEM as PAL_SYSTEM, run_code  # noqa: E402
from huggingface_hub import hf_hub_download
import pandas as pd
from llama_cpp import Llama

def gold_of(a): return float(a.split("####")[-1].strip().replace(",", ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--n_bank", type=int, default=400); ap.add_argument("--bank_seed", type=int, default=77)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--offset", type=int, default=300)
    ap.add_argument("--k_demo", type=int, default=2); ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--bank", default="bank.jsonl"); ap.add_argument("--out", default="rows.jsonl")
    a = ap.parse_args()
    llm = Llama(model_path=hf_hub_download(a.repo, a.file), n_ctx=3072, n_threads=os.cpu_count(), verbose=False, seed=0)
    def chat(msgs):
        r = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=a.max_tokens, seed=0)
        return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], r["usage"]["prompt_tokens"]

    # Phase 1: bank
    tr = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset"))
    bidx = list(range(len(tr))); random.Random(a.bank_seed).shuffle(bidx); bidx = bidx[:a.n_bank]
    done = {json.loads(l)["idx"] for l in open(a.bank)} if os.path.exists(a.bank) else set()
    with open(a.bank, "a") as f:
        for i in bidx:
            if i in done: continue
            t = time.time(); q = tr.question[i]; g = gold_of(tr.answer[i])
            cot, _, _ = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}])
            pal, _, _ = chat([{"role": "system", "content": PAL_SYSTEM}, {"role": "user", "content": q}])
            cp = extract(cot); pp = run_code(pal)[3]
            agree = cp is not None and pp is not None and abs(cp - pp) < 1e-4
            f.write(json.dumps({"idx": int(i), "q": q, "cot": cot, "cot_pred": cp, "pal_pred": pp, "agree": agree,
                                "gold_ok": cp is not None and abs(cp - g) < 1e-6}) + "\n"); f.flush()
            print(f"bank {i} agree={agree} {time.time()-t:.1f}s", flush=True)
    bank = [json.loads(l) for l in open(a.bank)]
    if len(bank) < a.n_bank:
        print("bank incomplete"); return
    mem = [b for b in bank if b["agree"]]
    print(f"bank n={len(bank)} kept={len(mem)} precision={sum(b['gold_ok'] for b in mem)/max(len(mem),1):.3f}", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit([b["q"] for b in mem])
    M = vec.transform([b["q"] for b in mem])

    # Phase 2: eval
    te = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    idx = list(range(len(te))); random.Random(1234).shuffle(idx); idx = idx[a.offset:a.offset + a.n]
    done = {json.loads(l)["idx"] for l in open(a.out)} if os.path.exists(a.out) else set()
    with open(a.out, "a") as f:
        for i in idx:
            if i in done: continue
            q = te.question[i]; g = gold_of(te.answer[i]); t = time.time()
            sims = (M @ vec.transform([q]).T).toarray().ravel()
            near = list(sims.argsort()[::-1][:a.k_demo])[::-1]  # most similar demo placed last (closest to query)
            rnd = random.Random(1000003 + int(i)).sample(range(len(mem)), a.k_demo)
            rec = {"idx": int(i), "gold": g, "near": [int(mem[j]["idx"]) for j in near],
                   "near_sim": [round(float(sims[j]), 3) for j in near], "rand": [int(mem[j]["idx"]) for j in rnd]}
            for arm, demos in (("Z", []), ("R", near), ("X", rnd)):
                msgs = [{"role": "system", "content": SYSTEM}]
                for j in demos:
                    msgs += [{"role": "user", "content": mem[j]["q"]}, {"role": "assistant", "content": mem[j]["cot"]}]
                text, ntok, ptok = chat(msgs + [{"role": "user", "content": q}])
                p = extract(text)
                rec[arm] = {"pred": p, "ok": p is not None and abs(p - g) < 1e-6, "tok": ntok, "ptok": ptok,
                            "hash": "####" in text, "capped": ntok >= a.max_tokens, "text": text}
            rec["secs"] = round(time.time() - t, 1)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(i, {k: rec[k]["ok"] for k in "ZRX"}, rec["secs"], flush=True)

if __name__ == "__main__":
    main()
