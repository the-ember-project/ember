"""M3 run 3: label-free DPO (CoT+PAL vote picks chosen/rejected CoT) on Kaggle GPU. Driver: `python run.py` runs phases in subprocesses
(so vLLM frees GPU memory between phases). See README for design. Smoke test on CPU:
`python run.py --phase gen --model Qwen/Qwen2.5-0.5B-Instruct --n_train 4 --backend hf --max_new 64 --k_cot 3 --k_pal 1` etc.
"""
import argparse, gzip, json, math, os, random, re, subprocess, sys, tempfile, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_COT = ("You are a careful math solver. Reason step by step, then give the final "
           "answer on its own line in the form '#### <number>'.")
SYS_PAL = ("You are a careful math solver. Write a short Python program that computes the answer to the "
           "problem. Use clear variable names and comments for each step. Store the final numeric answer "
           "in a variable named `answer` and end with `print(answer)`. Output only the code in one "
           "```python block.")
ARMS = ["MIXR", "COT"]

def extract(text):  # identical to gsm8k-baseline/eval.py
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if m: s = m[-1]
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums: return None
        s = nums[-1]
    try: return float(s.replace(",", ""))
    except ValueError: return None

def run_code(text, timeout=10):  # identical logic to gsm8k-pal/pal.py
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[0] if m else text
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        out = p.stdout.strip()
    except subprocess.TimeoutExpired:
        out = ""
    os.unlink(path)
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", out)
    try: return float(nums[-1].replace(",", "")) if nums else None
    except ValueError: return None

def same(a, b): return a is not None and b is not None and abs(a - b) < 1e-6
def key(v): return None if v is None else round(v, 6)

def gold_of(ans): return float(ans.split("####")[-1].strip().replace(",", ""))

def load_gsm8k(split):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    return pd.read_parquet(hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset"))

def sign_p(f, b):
    n = f + b
    if n == 0: return 1.0
    k = min(f, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

# ---------------- generation backends ----------------
class Gen:
    def __init__(self, a, lora_paths=None):
        self.a = a
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
        if a.backend == "vllm":
            from vllm import LLM
            kw = dict(enable_lora=True, max_lora_rank=16, max_loras=1) if lora_paths else {}
            self.llm = LLM(a.model, dtype="half", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed, **kw)
        else:
            import torch
            from transformers import AutoModelForCausalLM
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.dev = dev
            self.base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
            self.model = self.base

    def set_lora(self, path):
        if self.a.backend == "vllm": self.lora = path; return
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base, path).merge_and_unload() if path else self.base

    def chat(self, sys_, q):
        return self.tok.apply_chat_template([{"role": "system", "content": sys_}, {"role": "user", "content": q}],
                                            tokenize=False, add_generation_prompt=True)

    def __call__(self, prompts, n, temp, lora_id=0):
        a = self.a
        if a.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(n=n, temperature=temp, max_tokens=a.max_new)
            kw = {}
            if getattr(self, "lora", None):
                from vllm.lora.request import LoRARequest
                kw["lora_request"] = LoRARequest(f"a{lora_id}", lora_id, self.lora)
            outs = self.llm.generate(prompts, sp, **kw)
            return [[c.text for c in o.outputs] for o in outs], sum(len(c.token_ids) for o in outs for c in o.outputs)
        import torch
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res, ntok = [None] * len(prompts), 0
        bs = max(1, a.bs // n)
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            enc = self.tok([prompts[i] for i in ids], return_tensors="pt", padding=True).to(self.dev)
            kw = dict(do_sample=True, temperature=temp, top_p=1.0) if temp > 0 else dict(do_sample=False)
            with torch.no_grad():
                o = self.model.generate(**enc, max_new_tokens=a.max_new, num_return_sequences=n,
                                        pad_token_id=self.tok.eos_token_id, **kw)
            o = o[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(o, skip_special_tokens=True)
            ntok += int((o != self.tok.eos_token_id).sum())
            for j, i in enumerate(ids): res[i] = texts[j * n:(j + 1) * n]
        return res, ntok

# ---------------- phases ----------------
BBH_COT = ("Think step by step, keeping track of the full state after every step. "
           "End with a line of the form 'So the answer is X.' where X is the option letter in parentheses, e.g. (A).")

def parse_bbh(text):  # identical to bbh-mixvote2 parse_cot (mc branch)
    m = re.findall(r"answer is:?\s*\**\s*\(?([A-G])\)?(?![a-zA-Z])", text)
    if m: return f"({m[-1]})"
    m = re.findall(r"\(([A-G])\)", text)
    return f"({m[-1]})" if m else None

def phase_gen(a):
    df = load_gsm8k("train")
    idx = list(range(len(df))); random.Random(2029).shuffle(idx); idx = idx[:a.n_train]
    qs = [df.question[i] for i in idx]
    g = Gen(a); t0 = time.time()
    cp = [g.chat(SYS_COT, q) for q in qs]; pp = [g.chat(SYS_PAL, q) for q in qs]
    cg, n1 = g(cp, 1, 0.0); cs, n2 = g(cp, a.k_cot, 0.8)
    print(f"cot gen done {time.time()-t0:.0f}s tok={n1+n2}", flush=True)
    pg, n3 = g(pp, 1, 0.0); ps, n4 = g(pp, a.k_pal, 0.8)
    print(f"pal gen done {time.time()-t0:.0f}s tok={n3+n4}", flush=True)
    pal_texts = [t for i in range(len(qs)) for t in pg[i] + ps[i]]
    with ThreadPoolExecutor(8) as ex: pal_ans = list(ex.map(run_code, pal_texts))
    m = 1 + a.k_pal; rows = []
    for j, i in enumerate(idx):
        ct = cg[j] + cs[j]
        rows.append(dict(idx=i, gold=gold_of(df.answer[i]), cot_texts=ct, cot=[extract(t) for t in ct],
                         pal=pal_ans[j * m:(j + 1) * m]))
    json.dump(rows, open(os.path.join(a.work, "gen.json"), "w"))
    with gzip.open(os.path.join(a.out, "gen_answers.jsonl.gz"), "wt") as f:
        for r in rows: f.write(json.dumps({k: r[k] for k in ("idx", "gold", "cot", "pal")}) + "\n")
    print(f"gen phase {time.time()-t0:.0f}s tokens={n1+n2+n3+n4}", flush=True)

def label_of(arm, r):
    ca, pa = r["cot"], r["pal"]
    if arm == "GOLD": return r["gold"]
    if arm == "COT":
        c = Counter(key(x) for x in ca if x is not None)
        if not c: return None
        lab, v = c.most_common(1)[0]
        return lab if v >= 5 else None                      # >=5 of 8 CoT
    c = Counter(key(x) for x in ca + pa if x is not None)
    if not c: return None
    lab, v = c.most_common(1)[0]                            # MIXR: plurality >=5 of 13, both formats present
    return lab if v >= 5 and any(same(x, lab) for x in ca) and any(same(x, lab) for x in pa) else None

def build_pairs(rows, cap):
    rng = random.Random(2029); sets = {k: [] for k in ARMS}; stats = {}
    for arm in ARMS:
        nq = correct = 0
        for r in rows:
            lab = label_of(arm, r)
            if lab is None: continue
            ch = list(dict.fromkeys(t for t, x in zip(r["cot_texts"], r["cot"]) if same(x, lab)))
            rj = list(dict.fromkeys(t for t, x in zip(r["cot_texts"], r["cot"]) if not same(x, lab)))
            if not ch or not rj: continue
            nq += 1; correct += same(lab, r["gold"])
            rng.shuffle(ch); rng.shuffle(rj)
            sets[arm] += [(r["idx"], c, j) for c, j in list(zip(ch, rj))[:2]]
        stats[arm] = dict(pair_questions=nq, label_precision_on_pair_q=correct / max(1, nq), pairs=len(sets[arm]),
                          chosen_correct=sum(same(extract(c), gm) for (i, c, j) in sets[arm] for gm in [gold_map[i]]) / max(1, len(sets[arm])))
    E = min(cap, min(len(v) for v in sets.values()))
    for arm in ARMS: sets[arm] = rng.sample(sets[arm], E)
    stats["E"] = E
    return sets, stats

def seq_logp(model, ids, att, lab):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]; mask = tgt != -100
    lp = torch.gather(logits.log_softmax(-1), 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1), mask.sum(-1)

def phase_train(a):
    global torch, gold_map
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    try:
        import peft.import_utils as _iu, peft.tuners.lora.torchao as _lt
        _iu.is_torchao_available = lambda: False; _lt.is_torchao_available = lambda: False
    except Exception as e: print("torchao patch skipped:", e, flush=True)
    rows = json.load(open(os.path.join(a.work, "gen.json")))
    gold_map = {r["idx"]: r["gold"] for r in rows}
    df = load_gsm8k("train"); qmap = {r["idx"]: df.question[r["idx"]] for r in rows}
    sets, stats = build_pairs(rows, a.cap)
    json.dump(stats, open(os.path.join(a.out, "train_stats.json"), "w"), indent=1)
    print("train stats", json.dumps(stats), flush=True)
    cuda = torch.cuda.is_available(); dev = "cuda" if cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    def enc(i, t):
        msgs = [{"role": "system", "content": SYS_COT}, {"role": "user", "content": qmap[i]}]
        pr = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": t}], tokenize=False)
        pi, fi = tok(pr)["input_ids"], tok(full)["input_ids"][:a.max_len]
        return fi, [-100] * len(pi) + fi[len(pi):]
    log = {}
    for arm in (a.arms.split(",") if a.arms else ARMS):
        t0 = time.time(); torch.manual_seed(2029)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if cuda else torch.float32).to(dev)
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad: p.data = p.data.float()
        data = [(enc(i, c), enc(i, j)) for i, c, j in sets[arm]]
        if not data: print(f"{arm}: no data, skipped", flush=True); continue
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        mb, acc = a.mb, max(1, 16 // a.mb)
        total = math.ceil(len(data) * a.epochs / (mb * acc)); warm = max(1, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=cuda); step = 0; rng = random.Random(2029); hist = []
        model.train()
        for ep in range(a.epochs):
            order = list(range(len(data))); rng.shuffle(order)
            for s in range(0, len(order), mb):
                seqs = [data[j][0] for j in order[s:s + mb]] + [data[j][1] for j in order[s:s + mb]]
                n = len(seqs) // 2; L = max(len(x[0]) for x in seqs)
                ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in seqs], device=dev)
                lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in seqs], device=dev)
                att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in seqs], device=dev)
                with torch.no_grad(), model.disable_adapter(), torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    ref, _ = seq_logp(model, ids, att, lab)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    pol, ntok = seq_logp(model, ids, att, lab)
                margin = a.beta * ((pol[:n] - ref[:n]) - (pol[n:] - ref[n:]))
                dpo = -torch.nn.functional.logsigmoid(margin).mean()
                nll = -(pol[:n] / ntok[:n].clamp(min=1)).mean()
                loss = (dpo + a.alpha * nll) / acc
                scaler.scale(loss).backward()
                hist.append((dpo.item(), nll.item(), (margin > 0).float().mean().item(), margin.mean().item() / a.beta))
                if (s // mb + 1) % acc == 0 or s + mb >= len(order):
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1
                    if step % 10 == 0:
                        h = hist[-10 * acc:]; m4 = [sum(x[k] for x in h) / len(h) for k in range(4)]
                        print(f"{arm} step {step}/{total} dpo {m4[0]:.4f} nll {m4[1]:.4f} acc {m4[2]:.3f} logratio_margin {m4[3]:.3f}", flush=True)
        model.save_pretrained(os.path.join(a.work, f"lora_{arm}"))
        h0, h1 = hist[:40], hist[-40:]
        log[arm] = dict(n=len(data), steps=step, secs=round(time.time() - t0),
                        first=[round(sum(x[k] for x in h0) / len(h0), 4) for k in range(4)],
                        last=[round(sum(x[k] for x in h1) / len(h1), 4) for k in range(4)])
        print(f"trained {arm}: {json.dumps(log[arm])} (dpo, nll, pair-acc, logratio margin)", flush=True)
        del model, opt; cuda and torch.cuda.empty_cache()
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)

def paired(rx, ry):
    f = sum(p and not q for p, q in zip(rx, ry)); b = sum(q and not p for p, q in zip(rx, ry))
    return dict(diff=round(100 * (sum(rx) - sum(ry)) / len(rx), 2), fixed=f, broken=b, p=round(sign_p(f, b), 4))

def phase_eval(a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = load_gsm8k("train")                                    # held-out GSM8K-train questions: never used in gen/train
    idx = list(range(len(df))); random.Random(2029).shuffle(idx); idx = idx[3000:3000 + a.n_test]
    assert a.n_train is None or a.n_train <= 3000
    golds = [gold_of(df.answer[i]) for i in idx]
    bb = pd.read_parquet(hf_hub_download("lukaemon/bbh", "date_understanding/test-00000-of-00001.parquet", repo_type="dataset"))
    bb = bb.iloc[:a.n_bbh]
    arms = ["BASE"] + [x for x in ARMS if os.path.exists(os.path.join(a.work, f"lora_{x}"))]
    g = Gen(a, lora_paths=True); prompts = [g.chat(SYS_COT, df.question[i]) for i in idx]
    bprompts = [g.tok.apply_chat_template([{"role": "user", "content": q + "\n\n" + BBH_COT}], tokenize=False, add_generation_prompt=True)
                for q in bb["input"]]
    res, bres = {}, {}
    for j, arm in enumerate(arms):
        t0 = time.time()
        g.set_lora(None if arm == "BASE" else os.path.join(a.work, f"lora_{arm}"))
        outs, ntok = g(prompts, 1, 0.0, lora_id=j + 1)
        preds = [extract(o[0]) for o in outs]
        res[arm] = [same(p, gg) for p, gg in zip(preds, golds)]
        bouts, _ = g(bprompts, 1, 0.0, lora_id=j + 1)
        bpred = [parse_bbh(o[0]) for o in bouts]
        bres[arm] = [p == t for p, t in zip(bpred, bb["target"])]
        with open(os.path.join(a.out, f"eval_{arm}.jsonl"), "w") as f:
            for i, p, o, c in zip(idx, preds, outs, res[arm]):
                f.write(json.dumps(dict(idx=i, pred=p, correct=c, len=len(o[0]), tail=o[0][-300:])) + "\n")
            for i, p, o, c in zip(range(len(bb)), bpred, bouts, bres[arm]):
                f.write(json.dumps(dict(bbh_idx=i, pred=p, correct=c, tail=o[0][-300:])) + "\n")
        print(f"eval {arm}: gsm8k={sum(res[arm])/len(idx):.4f} bbh_date={sum(bres[arm])/len(bb):.3f} "
              f"tok/q={ntok/len(idx):.0f} {time.time()-t0:.0f}s", flush=True)
        if a.backend == "hf" and arm != "BASE":
            del g; g = Gen(a, lora_paths=True)
    summ = {}
    for x, y in [("MIX", "BASE"), ("MIX", "COT"), ("GOLD", "BASE"), ("COT", "BASE")]:
        if x in res and y in res:
            summ[f"{x}-{y}"] = paired(res[x], res[y]); summ[f"bbh_date {x}-{y}"] = paired(bres[x], bres[y])
    summ["acc"] = {k: round(sum(v) / len(idx), 4) for k, v in res.items()}
    summ["bbh_date_acc"] = {k: round(sum(v) / len(bb), 4) for k, v in bres.items()}
    summ["n"] = dict(gsm8k=len(idx), bbh_date=len(bb))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_train", type=int, default=None); ap.add_argument("--n_test", type=int, default=1500)
    ap.add_argument("--n_bbh", type=int, default=250); ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k_cot", type=int, default=7); ap.add_argument("--k_pal", type=int, default=4); ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=3031)
    ap.add_argument("--cap", type=int, default=2000); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=768); ap.add_argument("--arms", default="")
    ap.add_argument("--work", default="/tmp/m3dpo2"); ap.add_argument("--out", default=HERE)
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    if a.phase != "all":
        {"gen": phase_gen, "train": phase_train, "eval": phase_eval}[a.phase](a); return
    if a.backend == "auto":
        a.backend = "hf"
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "peft"], timeout=1200, check=True)
            subprocess.run([sys.executable, "-c", "from vllm import LLM, SamplingParams; "
                            f"l=LLM('{a.model}', dtype='half', max_model_len=1024, gpu_memory_utilization=0.5); "
                            "print('VLLM_OK', l.generate(['2+2='], SamplingParams(max_tokens=8))[0].outputs[0].text)"],
                           timeout=900, check=True)
            a.backend = "vllm"
        except Exception as e:
            print("vLLM unavailable, falling back to HF:", repr(e)[:300], flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
    if a.n_train is None: a.n_train = 3000 if a.backend == "vllm" else 1200
    print(f"backend={a.backend} n_train={a.n_train}", flush=True)
    argv, skip = [], False
    for x in sys.argv[1:]:
        if skip: skip = False; continue
        if x in ("--backend", "--n_train"): skip = True; continue
        argv.append(x)
    base = [sys.executable, os.path.abspath(__file__)] + argv
    common = ["--backend", a.backend, "--n_train", str(a.n_train)]
    # Smoke pass of every phase on a tiny slice first, so a late-phase failure costs minutes, not the 45-min gen.
    smoke = ["--n_test", "4", "--n_bbh", "4", "--cap", "4", "--epochs", "1", "--work", "/tmp/m3smoke", "--out", "/tmp/m3smoke_out"]
    for ph in ["gen", "train", "eval"]:
        t0 = time.time()
        r = subprocess.run(base + ["--phase", ph, "--backend", a.backend, "--n_train", "16"] + smoke)
        print(f"SMOKE {ph} exit={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0 and (a.backend == "hf" or ph == "train"):
            print("smoke failed; stopping before the full run"); sys.exit(r.returncode)
        if r.returncode != 0: print(f"smoke {ph} failed on vLLM; full run will fall back to HF there", flush=True); break
    for ph in ["gen", "train", "eval"]:
        t0 = time.time()
        r = subprocess.run(base + ["--phase", ph] + common)
        print(f"PHASE {ph} exit={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0 and a.backend == "vllm" and ph != "train":  # fall back to HF for this and later phases
            a.backend = "hf"
            if ph == "gen": a.n_train = 1200
            common = ["--backend", "hf", "--n_train", str(a.n_train)]
            print(f"vLLM phase {ph} failed; retrying with HF n_train={a.n_train}", flush=True)
            r = subprocess.run(base + ["--phase", ph] + common)
            print(f"PHASE {ph} (hf) exit={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0: sys.exit(r.returncode)
    print("DONE")

if __name__ == "__main__":
    main()
