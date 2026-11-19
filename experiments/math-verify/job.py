"""math-verify: 1.5B/3B best-of-8 on fresh MATH test items + small ORM scoring. See README.
Driver: `python job.py` (Kaggle). Every phase runs in its own process group with a timeout and hard exit.
CPU smoke: `python job.py --smoke_cpu` (HF generation, 0.5B policy, 0.6B RM, 2 items)."""
import argparse, gzip, json, math, os, random, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_COT = ("You are a careful math solver. Reason step by step, then give the final answer "
           "(a number: integer, decimal or fraction) in the form \\boxed{answer}.")
SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
POLICY = {"1.5B": "Qwen/Qwen2.5-1.5B-Instruct", "3B": "Qwen/Qwen2.5-3B-Instruct", "0.5B": "Qwen/Qwen2.5-0.5B-Instruct"}
RMS = {"rm17": "Skywork/Skywork-Reward-V2-Qwen3-1.7B", "rm06": "Skywork/Skywork-Reward-V2-Qwen3-0.6B"}

def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return None
    j = k = i + 7; d = 1
    while k < len(s) and d:
        d += (s[k] == "{") - (s[k] == "}"); k += 1
    return s[j:k - 1] if d == 0 else None

def num(b):  # total: never raises, rejects non-finite (same rules as m3-math-stack)
    try:
        v = _num(b)
        return v if v is None or math.isfinite(v) else None
    except Exception: return None

def _num(b):
    if b is None: return None
    b = b.strip()
    for t in ["\\!", "\\$", "$", ",", " ", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\%", "%", "\\text{ }"]: b = b.replace(t, "")
    m = re.fullmatch(r"(-?)\\[dt]?frac\{?(-?\d+)\}?\{?(-?\d+)\}?", b)
    if m: return (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3)) if int(m.group(3)) else None
    m = re.fullmatch(r"\(?(-?\d+)\)?/\(?(\d+)\)?", b)
    if m: return int(m.group(1)) / int(m.group(2)) if int(m.group(2)) else None
    if re.fullmatch(r"-?\d*\.?\d+", b): return float(b)
    return None

def extract(text):  # boxed number first; else last number (m3-math-stack rule)
    try:
        v = num(boxed(text))
        if v is not None: return v
        m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
        s = m[-1] if m else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
        if s is None: return None
        v = float(s.replace(",", "")); return v if math.isfinite(v) else None
    except Exception: return None

def load_items(off, n):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    dfs = []
    for sj in SUBJ:
        d = pd.read_parquet(hf_hub_download("EleutherAI/hendrycks_math", f"{sj}/test-00000-of-00001.parquet", repo_type="dataset"))
        d["subject"] = sj; dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["gold"] = df.solution.map(lambda x: num(boxed(x)))
    df = df[df.gold.notna()].reset_index(drop=True)
    idx = list(range(len(df))); random.Random(2029).shuffle(idx)
    return [dict(pos=off + j, idx=i, problem=df.problem[i], gold=float(df.gold[i]), level=df.level[i], subject=df.subject[i])
            for j, i in enumerate(idx[off:off + n])]

def phase_gen(a):
    items = load_items(a.off, a.n)
    model = POLICY[a.size]
    msgs = [[{"role": "system", "content": SYS_COT}, {"role": "user", "content": it["problem"]}] for it in items]
    outs = []  # per item: list of (text, n_tokens), greedy first
    if a.backend == "vllm":
        from vllm import LLM, SamplingParams
        llm = LLM(model, dtype="half", gpu_memory_utilization=0.85, max_model_len=a.max_new + 1024, seed=a.seed)
        g = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=a.max_new))
        s = llm.chat(msgs, SamplingParams(temperature=0.7, top_p=1.0, n=a.k - 1, max_tokens=a.max_new, seed=a.seed))
        for go, so in zip(g, s):
            outs.append([(go.outputs[0].text, len(go.outputs[0].token_ids))] + [(o.text, len(o.token_ids)) for o in so.outputs])
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model); m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        torch.manual_seed(a.seed)
        for ms in msgs:
            ids = tok.apply_chat_template(ms, add_generation_prompt=True, return_tensors="pt", return_dict=True)
            row = []
            for j in range(a.k):
                kw = dict(do_sample=False) if j == 0 else dict(do_sample=True, temperature=0.7, top_p=1.0)
                o = m.generate(**ids, max_new_tokens=a.max_new, **kw)[0][ids["input_ids"].shape[1]:]
                row.append((tok.decode(o, skip_special_tokens=True), int(len(o))))
            outs.append(row)
    with gzip.open(os.path.join(a.out, f"gen_{a.size}.jsonl.gz"), "wt") as f:
        for it, row in zip(items, outs):
            f.write(json.dumps(dict(it, texts=[t for t, _ in row], toks=[n for _, n in row],
                                    ans=[extract(t) for t, _ in row], boxed=[boxed(t) is not None for t, _ in row])) + "\n")
    print("gen done", a.size, len(outs), flush=True)

def phase_rm(a):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = [json.loads(l) for l in gzip.open(os.path.join(a.out, f"gen_{a.size}.jsonl.gz"), "rt")]
    cuda = torch.cuda.is_available(); dev = "cuda" if cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(RMS[a.rm]); tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    def load(dt): return AutoModelForSequenceClassification.from_pretrained(RMS[a.rm], dtype=dt, num_labels=1).to(dev).eval()
    jobs = []
    for r in rows:
        for j, t in enumerate(r["texts"]):
            s = tok.apply_chat_template([{"role": "user", "content": r["problem"]}, {"role": "assistant", "content": t}], tokenize=False)
            jobs.append((r["idx"], j, s))
    def score(model, subset, bs):
        res = {}
        subset = sorted(subset, key=lambda x: len(x[2]))
        for b in range(0, len(subset), bs):
            ch = subset[b:b + bs]
            enc = tok([c[2] for c in ch], return_tensors="pt", padding=True, truncation=True, max_length=3584, add_special_tokens=False).to(dev)
            with torch.no_grad(): lg = model(**enc).logits[:, 0].float().cpu().tolist()
            for c, v in zip(ch, lg): res[(c[0], c[1])] = v
        return res
    t0 = time.time(); m = load(torch.float16 if cuda else torch.float32)
    res = score(m, jobs, a.rm_bs)
    bad = [j for j in jobs if not math.isfinite(res[(j[0], j[1])])]
    print(f"rm {a.rm} {a.size}: {len(jobs)} scored in {time.time()-t0:.0f}s, non-finite {len(bad)}", flush=True)
    if bad:
        del m; torch.cuda.empty_cache() if cuda else None
        res.update(score(load(torch.float32), bad, 2))
    nonfin = sum(not math.isfinite(v) for v in res.values())
    out = {"n_nonfinite_fp16": len(bad), "n_nonfinite_final": nonfin,
           "scores": {str(r["idx"]): [res[(r["idx"], j)] for j in range(len(r["texts"]))] for r in rows}}
    json.dump(out, open(os.path.join(a.out, f"rm_{a.size}_{a.rm}.json"), "w"))

def run_phase(argv, tmo):
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--one"] + argv, start_new_session=True)
    try: rc = p.wait(timeout=tmo)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = "timeout"
    print("PHASE", " ".join(argv), "rc", rc, flush=True); return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true"); ap.add_argument("--smoke_cpu", action="store_true")
    ap.add_argument("--phase"); ap.add_argument("--size", default="1.5B"); ap.add_argument("--rm", default="rm17")
    ap.add_argument("--off", type=int, default=3700); ap.add_argument("--n", type=int, default=317)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=4242); ap.add_argument("--backend", default="vllm")
    ap.add_argument("--rm_bs", type=int, default=8); ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.one:
        rc = 0
        try: {"gen": phase_gen, "rm": phase_rm}[a.phase](a)
        except BaseException:
            import traceback; traceback.print_exc(); rc = 1
        sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
    if a.smoke_cpu:
        o = ["--out", "/tmp/mv_smoke", "--backend", "hf", "--n", "2", "--k", "2", "--max_new", "48"]
        run_phase(["--phase", "gen", "--size", "0.5B"] + o, 900); run_phase(["--phase", "rm", "--size", "0.5B", "--rm", "rm06"] + o, 900)
        return
    try: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], timeout=1500, check=True)
    except Exception as e: print("pip failed:", repr(e)[:300], flush=True)
    os.system(f"{sys.executable} -m pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers) '")
    # in-job smoke on a tiny slice: every phase, every model
    sm = ["--out", "/tmp/smoke", "--n", "2", "--k", "2", "--max_new", "128"]
    for size in ("1.5B", "3B"):
        if run_phase(["--phase", "gen", "--size", size] + sm, 900) != 0: print("SMOKE FAILED gen", size); return
    for rm in ("rm17", "rm06"):
        if run_phase(["--phase", "rm", "--size", "1.5B", "--rm", rm] + sm, 900) != 0: print("SMOKE FAILED rm", rm); return
    print("SMOKE OK", flush=True)
    full = ["--out", a.out, "--off", str(a.off), "--n", str(a.n)]
    plan = [(["--phase", "gen", "--size", "1.5B"], 1800), (["--phase", "rm", "--size", "1.5B", "--rm", "rm17"], 1500),
            (["--phase", "gen", "--size", "3B"], 2700), (["--phase", "rm", "--size", "3B", "--rm", "rm17"], 1500),
            (["--phase", "rm", "--size", "1.5B", "--rm", "rm06"], 900)]
    for argv, tmo in plan:
        if run_phase(argv + full, tmo) != 0 and run_phase(argv + full, tmo) != 0 and argv[1] == "gen":
            print("gen failed twice; skipping dependent phases", flush=True)
            if argv[3] == "1.5B": return
            break
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
