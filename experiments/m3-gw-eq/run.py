"""m3-gw-eq: equal-data control for m3-gw-iter round 2 (R2EQ: R1 teacher vs R1X: base teacher). See README.
Driver `python run.py` (Kaggle): installs vLLM, smoke-runs every phase on a tiny slice, then the full run.
CPU smoke: python run.py --phase gen --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --n_train 3 --k_cot 2 --k_pal 2 --max_new 64 ...
"""
import argparse, gzip, json, math, os, random, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gw

GSM_SYS = ("You are a careful math solver. Reason step by step, then give the final "
           "answer on its own line in the form '#### <number>'.")

def msgs(mode, q):
    return [{"role": "system", "content": gw.SYS_COT if mode == "cot" else gw.SYS_PAL},
            {"role": "user", "content": gw.EX_Q}, {"role": "assistant", "content": gw.EX_COT if mode == "cot" else gw.EX_PAL},
            {"role": "user", "content": q}]

def gsm_extract(text):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    s = m[-1] if m else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    try: return float(s.replace(",", "")) if s else None
    except ValueError: return None

def sign_p(f, b):
    n = f + b
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)

def paired(rx, ry):
    f = sum(p and not q for p, q in zip(rx, ry)); b = sum(q and not p for p, q in zip(rx, ry))
    return dict(diff=round(100 * (sum(rx) - sum(ry)) / len(rx), 2), fixed=f, broken=b, p=round(sign_p(f, b), 4))

class Gen:
    def __init__(self, a, lora=False):
        self.a = a
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
        if a.backend == "vllm":
            from vllm import LLM
            kw = dict(enable_lora=True, max_lora_rank=16, max_loras=1) if lora else {}
            self.llm = LLM(a.model, dtype="half", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed, **kw)
        else:
            import torch
            from transformers import AutoModelForCausalLM
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev)
            self.model = self.base
        self.lora = None

    def set_lora(self, path):
        if self.a.backend == "vllm": self.lora = path; return
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base, path) if path else self.base

    def chat(self, m): return self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    def __call__(self, prompts, n, temp, lora_id=0):
        a = self.a
        if a.backend == "vllm":
            from vllm import SamplingParams
            kw = {}
            if self.lora:
                from vllm.lora.request import LoRARequest
                kw["lora_request"] = LoRARequest(f"a{lora_id}", lora_id, self.lora)
            outs = self.llm.generate(prompts, SamplingParams(n=n, temperature=temp, max_tokens=a.max_new), **kw)
            return [[c.text for c in o.outputs] for o in outs], [[len(c.token_ids) for c in o.outputs] for o in outs]
        import torch
        res, lens = [None] * len(prompts), [None] * len(prompts)
        for i, p in enumerate(prompts):
            enc = self.tok([p], return_tensors="pt").to(self.dev)
            kw = dict(do_sample=True, temperature=temp) if temp > 0 else dict(do_sample=False)
            with torch.no_grad():
                o = self.model.generate(**enc, max_new_tokens=a.max_new, num_return_sequences=n, pad_token_id=self.tok.eos_token_id, **kw)
            o = o[:, enc["input_ids"].shape[1]:]
            res[i] = self.tok.batch_decode(o, skip_special_tokens=True)
            lens[i] = [int((r != self.tok.eos_token_id).sum()) for r in o]
        return res, lens


def phase_gen(a, rnd):
    seed, kc = (9001, a.k_cot) if rnd == 1 else (9004, a.k_cot2)  # rnd 3: base model on the round-2 pool (control)
    tasks = [gw.make_task(i, 10, seed) for i in range(a.n_train)]
    g = Gen(a, lora=(rnd == 2)); t0 = time.time()
    if rnd == 2: g.set_lora(os.path.join(a.work, "lora_R1"))
    cs, _ = g([g.chat(msgs("cot", q)) for q, _ in tasks], kc, 0.8, lora_id=1)
    ps, _ = g([g.chat(msgs("pal", q)) for q, _ in tasks], a.k_pal, 0.8, lora_id=1)
    print(f"gen{rnd} done {time.time()-t0:.0f}s", flush=True)
    flat = [t for x in ps for t in x]
    with ThreadPoolExecutor(8) as ex: pal = [r[3] for r in ex.map(gw.run_code, flat)]
    rows = []
    for i, (q, gold) in enumerate(tasks):
        rows.append(dict(seed=seed, idx=i, gold=f"{gold[0]},{gold[1]}", cot_texts=cs[i], cot=[gw.parse_cot(t) for t in cs[i]],
                         pal=pal[i * a.k_pal:(i + 1) * a.k_pal]))
    json.dump(rows, open(os.path.join(a.work, f"gen{rnd}.json"), "w"))
    with gzip.open(os.path.join(a.out, f"gen{rnd}_answers.jsonl.gz"), "wt") as f:
        for r in rows: f.write(json.dumps({k: r[k] for k in ("seed", "idx", "gold", "cot", "pal")}) + "\n")
    cacc = sum(x == r["gold"] for r in rows for x in r["cot"]) / max(1, sum(len(r["cot"]) for r in rows))
    pacc = sum(x == r["gold"] for r in rows for x in r["pal"]) / max(1, sum(len(r["pal"]) for r in rows))
    print(f"gen{rnd}: cot sample acc {cacc:.3f} pal sample acc {pacc:.3f}", flush=True)

def plural(src, need):
    c = Counter(x for x in src if x is not None)
    if not c: return None
    lab, v = c.most_common(1)[0]
    return lab if v >= need else None

def build_set(rows, fake=False):
    rng = random.Random(9001); s, nl, ok, nj, okj = [], 0, 0, 0, 0
    for r in rows:
        j = plural(r["pal"] + r["cot"], 5)
        if j is not None: nj += 1; okj += j == r["gold"]
        lab = plural(r["pal"], 3)
        if fake: ch = list(dict.fromkeys(r["cot_texts"]))
        else:
            if lab is None: continue
            nl += 1; ok += lab == r["gold"]
            ch = list(dict.fromkeys(t for t, x in zip(r["cot_texts"], r["cot"]) if x == lab))
        rng.shuffle(ch); s += [(r["seed"], r["idx"], t) for t in ch[:2]]
    st = dict(tasks=len(rows), labelled_tasks=nl, label_precision=round(ok / max(1, nl), 4), traces=len(s),
              traces_correct=0,
              joint_labelled=nj, joint_precision=round(okj / max(1, nj), 4))
    gold = {(r["seed"], r["idx"]): r["gold"] for r in rows}
    st["traces_correct"] = round(sum(gw.parse_cot(t) == gold[(sd, i)] for sd, i, t in s) / max(1, len(s)), 4)
    return s, st
def phase_train(a, rnd):
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    try:
        import peft.import_utils as _iu, peft.tuners.lora.torchao as _lt
        _iu.is_torchao_available = lambda: False; _lt.is_torchao_available = lambda: False
    except Exception as e: print("torchao patch skipped:", e, flush=True)
    stats, S = {}, {}
    for r in ([1] if rnd == 1 else [1, 2, 3]):
        S[r], stats[f"gen{r}"] = build_set(json.load(open(os.path.join(a.work, f"gen{r}.json"))), a.fake)
    if rnd == 1:
        sets = {"R1": S[1]}; ARMS = ["R1"]
    else:  # equal-data control: same round-1 traces + equal-size round-2-pool traces from R1 teacher vs base teacher
        n2 = min(len(S[2]), len(S[3])); rs = random.Random(9002)
        s2 = rs.sample(S[2], n2); s3 = rs.sample(S[3], n2)
        sets = {"R2EQ": S[1] + s2, "R1X": S[1] + s3}; ARMS = ["R2EQ", "R1X"]
        stats["n_pool2_traces_each"] = n2
    json.dump(stats, open(os.path.join(a.out, f"train_stats_{'R1' if rnd == 1 else 'EQ'}.json"), "w"), indent=1)
    print("train stats", json.dumps(stats), flush=True)
    cuda = torch.cuda.is_available(); dev = "cuda" if cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    def enc(sd, i, t):
        m = msgs("cot", gw.make_task(i, 10, sd)[0]); pr = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(m + [{"role": "assistant", "content": t}], tokenize=False)
        pi, fi = tok(pr)["input_ids"], tok(full)["input_ids"][:a.max_len]
        return fi, [-100] * len(pi) + fi[len(pi):]
    log = {}
    for arm in ARMS:
        data = [enc(sd, i, t) for sd, i, t in sets[arm]]
        if not data: print(f"{arm}: no data", flush=True); continue
        t0 = time.time(); torch.manual_seed(9001)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if cuda else torch.float32).to(dev)
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for p in model.parameters():
            if p.requires_grad: p.data = p.data.float()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        mb, acc = a.mb, max(1, 16 // a.mb)
        total = math.ceil(len(data) * a.epochs / (mb * acc)); warm = max(1, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=cuda); step = 0; rng = random.Random(9001); hist = []
        model.train()
        for ep in range(a.epochs):
            order = list(range(len(data))); rng.shuffle(order)
            for s in range(0, len(order), mb):
                seqs = [data[j] for j in order[s:s + mb]]; L = max(len(x[0]) for x in seqs)
                ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in seqs], device=dev)
                lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in seqs], device=dev)
                att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in seqs], device=dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
                scaler.scale(loss / acc).backward(); hist.append(loss.item())
                if (s // mb + 1) % acc == 0 or s + mb >= len(order):
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1
                    if step % 20 == 0: print(f"{arm} step {step}/{total} loss {sum(hist[-20*acc:])/len(hist[-20*acc:]):.4f}", flush=True)
        model.save_pretrained(os.path.join(a.work, f"lora_{arm}"))
        log[arm] = dict(n=len(data), steps=step, secs=round(time.time() - t0),
                        loss_first=round(sum(hist[:40]) / len(hist[:40]), 4), loss_last=round(sum(hist[-40:]) / len(hist[-40:]), 4))
        print(f"trained {arm}: {json.dumps(log[arm])}", flush=True)
        del model, opt; cuda and torch.cuda.empty_cache()
    json.dump(log, open(os.path.join(a.out, f"train_log_{'R1' if rnd == 1 else 'EQ'}.json"), "w"), indent=1)


def g3(cot, pals):
    c = Counter(x for x in [cot] + pals if x is not None)
    if not c: return None
    top = max(c.values()); 
    return cot if cot is not None and c[cot] == top else c.most_common(1)[0][0]

def phase_eval(a):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    ev = {"gw10": [gw.make_task(i, 10, 9005) for i in range(a.n_test)],
          "gw14": [gw.make_task(i, 14, 9006) for i in range(a.n_test14)]}
    gs = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    gidx = list(range(len(gs))); random.Random(1234).shuffle(gidx); gidx = gidx[:a.n_gsm]
    arms = ["BASE"] + [x for x in ["R2EQ", "R1X"] if os.path.exists(os.path.join(a.work, f"lora_{x}"))]
    g = Gen(a, lora=True)
    P = {k: [g.chat(msgs("cot", q)) for q, _ in v] for k, v in ev.items()}
    P["gsm"] = [g.chat([{"role": "system", "content": GSM_SYS}, {"role": "user", "content": gs.question[i]}]) for i in gidx]
    PP = [g.chat(msgs("pal", q)) for q, _ in ev["gw10"]]
    res = {}
    for j, arm in enumerate(arms):
        t0 = time.time(); g.set_lora(None if arm == "BASE" else os.path.join(a.work, f"lora_{arm}")); res[arm] = {}
        with open(os.path.join(a.out, f"eval_{arm}.jsonl"), "w") as f:
            for k in ["gw10", "gw14", "gsm"]:
                outs, lens = g(P[k], 1, 0.0, lora_id=j + 1)
                if k == "gsm":
                    golds = [float(gs.answer[i].split("####")[-1].strip().replace(",", "")) for i in gidx]
                    preds = [gsm_extract(o[0]) for o in outs]
                    ok = [p is not None and abs(p - gg) < 1e-6 for p, gg in zip(preds, golds)]; ids = gidx
                else:
                    golds = [f"{t[1][0]},{t[1][1]}" for t in ev[k]]
                    preds = [gw.parse_cot(o[0]) for o in outs]; ok = [p == gg for p, gg in zip(preds, golds)]; ids = range(len(golds))
                res[arm][k] = ok
                extra = [{}] * len(ok)
                if k == "gw10":
                    pouts, _ = g(PP, a.k_g3, 0.7, lora_id=j + 1)
                    flat = [t for x in pouts for t in x]
                    with ThreadPoolExecutor(8) as ex: pa = [r[3] for r in ex.map(gw.run_code, flat)]
                    pals = [pa[i * a.k_g3:(i + 1) * a.k_g3] for i in range(len(ok))]
                    gv = [g3(p, pl) for p, pl in zip(preds, pals)]
                    res[arm]["g3"] = [v == gg for v, gg in zip(gv, golds)]
                    extra = [dict(pal=pl, g3=v, g3_correct=v == gg) for pl, v, gg in zip(pals, gv, golds)]
                    print(f"eval {arm} G3: acc={sum(res[arm]['g3'])/len(ok):.4f}", flush=True)
                for i, p, gg, o, n, c, e in zip(ids, preds, golds, outs, lens, ok, extra):
                    f.write(json.dumps(dict(set=k, idx=i, gold=gg, pred=p, correct=c, ntok=n[0], text=o[0][-800:], **e)) + "\n")
                skip = sum(n[0] < 30 for n in lens) / len(lens)
                print(f"eval {arm} {k}: acc={sum(ok)/len(ok):.4f} skip<30tok={skip:.3f} n={len(ok)}", flush=True)
        print(f"eval {arm} done {time.time()-t0:.0f}s", flush=True)
        if a.backend == "hf": g = Gen(a, lora=True)
    summ = {"acc": {arm: {k: round(sum(v) / len(v), 4) for k, v in r.items()} for arm, r in res.items()}}
    for x, y, k in [("R2EQ", "R1X", "gw10"), ("R2EQ", "R1X", "gw14"), ("R2EQ", "R1X", "g3"), ("R1X", "BASE", "gw10"),
                    ("R2EQ", "BASE", "gw10"), ("R2EQ", "BASE", "gsm"), ("R1X", "BASE", "gsm")]:
        if x in res and y in res: summ[f"{k} {x}-{y}"] = paired(res[x][k], res[y][k])
    for x in ["R2EQ", "R1X"]:
        if x in res: summ[f"g3 {x}-BASE"] = paired(res[x]["g3"], res["BASE"]["g3"])
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all"); ap.add_argument("--backend", default="auto")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_train", type=int, default=1500); ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--n_test14", type=int, default=300); ap.add_argument("--n_gsm", type=int, default=300)
    ap.add_argument("--k_cot", type=int, default=8); ap.add_argument("--k_cot2", type=int, default=4); ap.add_argument("--k_g3", type=int, default=3); ap.add_argument("--fake", action="store_true"); ap.add_argument("--k_pal", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=700); ap.add_argument("--seed", type=int, default=5151)
    ap.add_argument("--cap", type=int, default=1500); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1280)
    ap.add_argument("--work", default="/tmp/m3eq"); ap.add_argument("--out", default=HERE)
    a = ap.parse_args(); os.makedirs(a.work, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    if a.phase != "all":
        {"gen1": lambda a: phase_gen(a, 1), "gen2": lambda a: phase_gen(a, 2), "train1": lambda a: phase_train(a, 1),
         "gen3": lambda a: phase_gen(a, 3), "train2": lambda a: phase_train(a, 2), "eval": phase_eval}[a.phase](a); return
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
            print("vLLM unavailable:", repr(e)[:300], flush=True); sys.exit(1)  # HF gen would be far too slow here
    base = [sys.executable, os.path.abspath(__file__), "--backend", a.backend]
    smoke = ["--n_train", "12", "--n_test", "4", "--n_test14", "2", "--n_gsm", "2", "--epochs", "1",
             "--fake", "--work", "/tmp/m3eq_smoke", "--out", "/tmp/m3eq_smoke_out"]
    for ph in ["gen1", "train1", "gen2", "gen3", "train2", "eval"]:
        t0 = time.time(); r = subprocess.run(base + ["--phase", ph] + smoke)
        print(f"SMOKE {ph} exit={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0: print("smoke failed; stopping"); sys.exit(r.returncode)
    for ph in ["gen1", "train1", "gen2", "gen3", "train2", "eval"]:
        t0 = time.time(); r = subprocess.run(base + ["--phase", ph])
        print(f"PHASE {ph} exit={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0: sys.exit(r.returncode)
    print("DONE")

if __name__ == "__main__":
    main()
