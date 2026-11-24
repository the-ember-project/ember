"""m3-grpo-orm (from m3-grpo-rep): GRPO + LoRA on Qwen2.5-1.5B-Instruct (MATH-train) with the Skywork-Reward-V2-Qwen3-0.6B
score as reward (no gold), 2 replicates; greedy 2048-token eval on MATH-test positions 0-999 for BASE/ORM1/ORM2/3B;
greedy+7 samples on positions 0-499 for ORM1 and BASE, RM-scored (WSC8, M5 test). See README.
Kaggle gives 2 T4s: arms train two at a time, one per GPU (CUDA_VISIBLE_DEVICES), each in its own process group with a
timeout and hard exit (vLLM hang lesson). Training config = m3-grpo pilot (69 s/step on one T4) except lr 3e-5 constant.
CPU smoke: python run.py --cpu_smoke
"""
import argparse, json, math, os, random, re, signal, subprocess, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_COT = ("You are a careful math solver. Reason step by step, then give the final answer "
           "(a number: integer, decimal or fraction) in the form \\boxed{answer}.")  # identical to m3-math-stack
SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
BBH_COT = ("Think step by step, keeping track of the full state after every step. "
           "End with a line of the form 'So the answer is X.' where X is the option letter in parentheses, e.g. (A).")
WORK = "/tmp/m3grpo"
RS = {"ORM1": ["--seed", "10", "--shuf", "34"], "ORM2": ["--seed", "11", "--shuf", "35"]}  # fresh training seed + prompt draw per replicate

def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return None
    j = k = i + 7; d = 1
    while k < len(s) and d:
        d += (s[k] == "{") - (s[k] == "}"); k += 1
    return s[j:k - 1] if d == 0 else None

def num(b):  # total: never raises, rejects non-finite (same rules as m3-math-stack)
    try:
        if b is None: return None
        b = b.strip()
        for t in ["\\!", "\\$", "$", ",", " ", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\%", "%", "\\text{ }"]: b = b.replace(t, "")
        m = re.fullmatch(r"(-?)\\[dt]?frac\{?(-?\d+)\}?\{?(-?\d+)\}?", b)
        if m: v = (-1 if m.group(1) else 1) * int(m.group(2)) / int(m.group(3)) if int(m.group(3)) else None
        else:
            m = re.fullmatch(r"\(?(-?\d+)\)?/\(?(\d+)\)?", b)
            if m: v = int(m.group(1)) / int(m.group(2)) if int(m.group(2)) else None
            elif re.fullmatch(r"-?\d*\.?\d+", b): v = float(b)
            else: v = None
        return v if v is None or math.isfinite(v) else None
    except Exception: return None

def extract(text):  # identical rule to m3-math-stack: boxed number first; else last number
    try:
        v = num(boxed(text))
        if v is not None: return v
        m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
        if m: s = m[-1]
        else:
            nums = re.findall(r"-?[\d,]*\.?\d+", text)
            if not nums: return None
            s = nums[-1]
        v = float(s.replace(",", ""))
        return v if math.isfinite(v) else None
    except Exception: return None

def same(a, b): return a is not None and b is not None and abs(a - b) <= 1e-4 * max(1.0, abs(b))
def key(v): return None if v is None else round(v, 4)
def text_of(c): return c[-1]["content"] if isinstance(c, list) else c

def load_math(split):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    dfs = []
    for sj in SUBJ:
        for t in range(3):
            try: d = pd.read_parquet(hf_hub_download("EleutherAI/hendrycks_math", f"{sj}/{split}-00000-of-00001.parquet", repo_type="dataset")); break
            except Exception:
                if t == 2: raise
                time.sleep(20)
        d["subject"] = sj; dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["gold"] = df.solution.map(lambda x: num(boxed(x)))
    return df[df.gold.notna()].reset_index(drop=True)

RM_ID = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
_RM = {}
def rm_scores(problems, texts, bs=8):  # same input format as math-verify: [user: problem, assistant: response]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not _RM:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(RM_ID); tok.padding_side = "right"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        m = AutoModelForSequenceClassification.from_pretrained(RM_ID, dtype=torch.float16 if dev == "cuda" else torch.float32,
                                                               num_labels=1).to(dev).eval()
        _RM.update(tok=tok, m=m, dev=dev)
    tok, m, dev = _RM["tok"], _RM["m"], _RM["dev"]
    strs = [tok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": t}], tokenize=False)
            for p, t in zip(problems, texts)]
    out = [float("nan")] * len(strs)
    order = sorted(range(len(strs)), key=lambda i: len(strs[i]))
    for b in range(0, len(order), bs):
        ch = order[b:b + bs]
        enc = tok([strs[i] for i in ch], return_tensors="pt", padding=True, truncation=True, max_length=3584,
                  add_special_tokens=False).to(dev)
        with torch.no_grad(): lg = m(**enc).logits[:, 0].float().cpu().tolist()
        for i, v in zip(ch, lg): out[i] = v
    return out

def auroc(pos, neg):
    if not pos or not neg: return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))

STATS = []
def make_reward(kind, seed=0):
    kind = re.sub(r"\d+$", "", kind)  # ORM1/ORM2 = ORM reward, separate replicates
    def reward(prompts, completions, gold, **kw):
        texts = [text_of(c) for c in completions]
        preds = [num(boxed(t)) for t in texts]
        gold_r = [1.0 if same(p, g) else 0.0 for p, g in zip(preds, gold)]
        if kind != "ORM": raise ValueError(kind)
        probs = [p[-1]["content"] if isinstance(p, list) else p for p in prompts]  # the user turn = problem
        sc = rm_scores(probs, texts)
        fin = [x for x in sc if math.isfinite(x)]
        nonfin = len(sc) - len(fin)
        sc = [x if math.isfinite(x) else (min(fin) if fin else 0.0) for x in sc]
        STATS.append(dict(gold_acc=sum(gold_r) / len(gold_r), boxed=sum(p is not None for p in preds) / len(preds),
                          rm_mean=sum(sc) / len(sc), rm_nonfinite=nonfin, mean_chars=sum(map(len, texts)) / len(texts),
                          rm_auroc=auroc([x for x, g in zip(sc, gold_r) if g], [x for x, g in zip(sc, gold_r) if not g])))
        return sc
    reward.__name__ = f"r_{kind}"
    return reward

def do_train(a):
    import torch, dataclasses, importlib
    from datasets import Dataset
    from peft import LoraConfig
    for mod in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):  # Kaggle torchao 0.10 clash
        try: setattr(importlib.import_module(mod), "is_torchao_available", lambda: False)
        except Exception as e: print("torchao patch skipped", mod, repr(e)[:120], flush=True)
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    df = load_math("train")
    idx = list(range(len(df))); random.Random(a.shuf).shuffle(idx); idx = idx[:a.steps * 4 + 8]
    ds = Dataset.from_list([dict(prompt=[{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}],
                                 gold=float(df.gold[i])) for i in idx])
    cuda = torch.cuda.is_available()
    want = dict(output_dir=f"{WORK}/ckpt_{a.arm}{a.tag}", learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0,
                per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga, num_generations=a.ng,
                max_completion_length=a.max_new, max_steps=a.steps, logging_steps=1, temperature=1.0, beta=0.0,
                use_vllm=bool(a.vllm), vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_util, vllm_max_model_length=1536,
                fp16=cuda, bf16=False, gradient_checkpointing=True, report_to="none", save_strategy="no", seed=a.seed,
                shuffle_dataset=False, model_init_kwargs={"torch_dtype": torch.float16 if cuda else torch.float32}, use_cpu=not cuda)
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in want if k not in fields]
    if dropped: print("GRPOConfig: dropped unknown args", dropped, flush=True)
    cfg = GRPOConfig(**{k: v for k, v in want.items() if k in fields})
    times, t_start = [], time.time()
    class T(TrainerCallback):
        def on_step_begin(self, *x, **k): self.t = time.time()
        def on_step_end(self, args, state, control, **k):
            times.append(time.time() - self.t); print(f"[{a.arm}] step {len(times)} {times[-1]:.1f}s", flush=True)
            if time.time() - t_start > a.max_train_min * 60: control.should_training_stop = True; print("time guard: stopping", flush=True)
    tr = GRPOTrainer(model=a.model, reward_funcs=[make_reward(a.arm, seed=a.seed)], args=cfg, train_dataset=ds,
                     peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                     callbacks=[T()])
    tr.train()
    tr.save_model(f"{WORK}/lora_{a.arm}{a.tag}")
    out = dict(arm=a.arm, steps=len(times), sec_per_step=times, total_s=round(time.time() - t_start), stats=STATS,
               log=tr.state.log_history, gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None)
    json.dump(out, open(os.path.join(a.out, f"{a.tag}train_{a.arm}.json"), "w"), default=str)
    print(f"TRAINED {a.arm} steps={len(times)}", flush=True)

def eval_items(n, off):
    df = load_math("test")
    idx = list(range(len(df))); random.Random(2029).shuffle(idx); return df, idx[off:off + n]

def make_gen(a, arms):
    """Returns gen(prompts, arm, j, sp_kw) -> list of lists of (text, n_tokens)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    lora = lambda arm: f"{WORK}/lora_{arm}{a.tag}"
    if a.vllm:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        use_lora = any(x != "BASE" for x in arms)
        llm = LLM(a.model, dtype="half", gpu_memory_utilization=0.85, max_model_len=a.eval_max_new + 640, seed=0,
                  **(dict(enable_lora=True, max_lora_rank=16, max_loras=1) if use_lora else {}))
        def gen(ps, arm, j, n=1, temp=0.0, seed=None):
            kw = {} if arm == "BASE" else dict(lora_request=LoRARequest(f"a{j}", j, lora(arm)))
            sp = SamplingParams(n=n, temperature=temp, top_p=1.0, max_tokens=a.eval_max_new, seed=seed)
            return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in llm.generate(ps, sp, **kw)]
    else:  # CPU smoke path only
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        def gen(ps, arm, j, n=1, temp=0.0, seed=None):
            m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
            if arm != "BASE": m = PeftModel.from_pretrained(m, lora(arm)).merge_and_unload()
            if seed is not None: torch.manual_seed(seed)
            res = []
            for p in ps:
                enc = tok(p, return_tensors="pt"); row = []
                for _ in range(n):
                    kw = dict(do_sample=False) if temp == 0 else dict(do_sample=True, temperature=temp, top_p=1.0)
                    o = m.generate(**enc, max_new_tokens=a.eval_max_new, **kw)[0, enc["input_ids"].shape[1]:]
                    row.append((tok.decode(o, skip_special_tokens=True), int(len(o))))
                res.append(row)
            return res
    return gen, chat, lora

def do_eval(a):  # greedy MATH (+ BBH-date drift); file name eval_{label or arm}
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df, idx = eval_items(a.n_test, a.test_off)
    bb = pd.read_parquet(hf_hub_download("lukaemon/bbh", "date_understanding/test-00000-of-00001.parquet", repo_type="dataset")).iloc[:a.n_bbh]
    arms = [x for x in a.arms.split(",") if x == "BASE" or os.path.isdir(f"{WORK}/lora_{x}{a.tag}")]
    print("eval arms present:", arms, "requested:", a.arms, flush=True)
    gen, chat, _ = make_gen(a, arms)
    prompts = [chat([{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}]) for i in idx]
    bprompts = [chat([{"role": "user", "content": q + "\n\n" + BBH_COT}]) for q in bb["input"]]
    for j, arm in enumerate(arms):
        t0 = time.time(); name = a.label or arm
        outs = [r[0][0] for r in gen(prompts, arm, j + 1)]
        bouts = [r[0][0] for r in gen(bprompts, arm, j + 1)] if len(bprompts) else []
        with open(os.path.join(a.out, f"{a.tag}eval_{name}.jsonl"), "w") as f:
            for i, o in zip(idx, outs):
                p = extract(o)
                f.write(json.dumps(dict(idx=i, subject=df.subject[i], level=df.level[i], gold=float(df.gold[i]), pred=p,
                                        correct=same(p, float(df.gold[i])), len=len(o), boxed=boxed(o) is not None, tail=o[-300:])) + "\n")
            for i, o in enumerate(bouts):
                m = re.findall(r"answer is:?\s*\**\s*\(?([A-G])\)?(?![a-zA-Z])", o)
                p = f"({m[-1]})" if m else None
                f.write(json.dumps(dict(bbh_idx=i, pred=p, correct=p == bb["target"].iloc[i], tail=o[-300:])) + "\n")
        acc = sum(same(extract(o), float(df.gold[i])) for i, o in zip(idx, outs)) / len(idx)
        print(f"EVAL {name}{a.tag}: math={acc:.4f} boxed={sum(boxed(o) is not None for o in outs)/len(outs):.3f} {time.time()-t0:.0f}s", flush=True)

def do_sample(a):  # greedy + 7 samples at T=0.7 (math-verify protocol); full texts to WORK, slim copy to out
    import gzip
    arm = a.arm
    if arm != "BASE" and not os.path.isdir(f"{WORK}/lora_{arm}{a.tag}"): raise SystemExit(f"no adapter for {arm}")
    df, idx = eval_items(a.n_samp, a.test_off)
    gen, chat, _ = make_gen(a, [arm])
    prompts = [chat([{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}]) for i in idx]
    t0 = time.time()
    g = gen(prompts, arm, 1); s = gen(prompts, arm, 1, n=a.k - 1, temp=0.7, seed=a.samp_seed)
    rows = []
    for i, go, so in zip(idx, g, s):
        row = go + so
        rows.append(dict(idx=i, problem=df.problem[i], gold=float(df.gold[i]), level=df.level[i], subject=df.subject[i],
                         texts=[t for t, _ in row], toks=[n for _, n in row], ans=[extract(t) for t, _ in row],
                         boxed=[boxed(t) is not None for t, _ in row]))
    with gzip.open(f"{WORK}/gen_{arm}{a.tag}.jsonl.gz", "wt") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with gzip.open(os.path.join(a.out, f"{a.tag}gen_{arm}.jsonl.gz"), "wt") as f:
        for r in rows: f.write(json.dumps(dict({k: v for k, v in r.items() if k not in ("problem", "texts")},
                                               tails=[t[-300:] for t in r["texts"]])) + "\n")
    print(f"SAMPLED {arm}{a.tag}: {len(rows)} items x {a.k} in {time.time()-t0:.0f}s", flush=True)

def do_rm(a):
    import gzip
    rows = [json.loads(l) for l in gzip.open(f"{WORK}/gen_{a.arm}{a.tag}.jsonl.gz", "rt")]
    t0 = time.time()
    flat = [(r["idx"], j, r["problem"], t) for r in rows for j, t in enumerate(r["texts"])]
    sc = rm_scores([x[2] for x in flat], [x[3] for x in flat])
    bad = [k for k, v in enumerate(sc) if not math.isfinite(v)]
    if bad:  # fp32 re-score of non-finite ones (math-verify rule)
        import torch
        _RM["m"] = _RM["m"].float()
        for k, v in zip(bad, rm_scores([flat[k][2] for k in bad], [flat[k][3] for k in bad], bs=2)): sc[k] = v
    res = {}
    for (i, j, _, _), v in zip(flat, sc): res.setdefault(str(i), []).append(v)
    json.dump({"n_nonfinite_fp16": len(bad), "n_nonfinite_final": sum(not math.isfinite(v) for v in sc), "scores": res},
              open(os.path.join(a.out, f"{a.tag}rm_{a.arm}_rm06.json"), "w"))
    print(f"RM {a.arm}{a.tag}: {len(flat)} scored in {time.time()-t0:.0f}s, non-finite fp16 {len(bad)}", flush=True)

def launch(args, gpu, tmo, log):
    env = dict(os.environ)
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["MASTER_PORT"] = str(29500 + 17 * (gpu + 1))
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args, start_new_session=True, env=env,
                         stdout=open(log, "a"), stderr=subprocess.STDOUT)
    p._tmo, p._t0, p._log, p._args = tmo, time.time(), log, args
    return p

def wait_all(ps):
    rcs = []
    for p in ps:
        try: rc = p.wait(timeout=max(1, p._tmo - (time.time() - p._t0)))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = "timeout"
        os.system(f"grep -E 'EVAL|TRAINED|SAMPLED|RM |Error|error|Traceback|time guard' {p._log} | tail -8; tail -3 {p._log}")
        print(f"PROC {' '.join(p._args[:4])} exit={rc} {time.time()-p._t0:.0f}s", flush=True); rcs.append(rc)
    return rcs

def chain(steps, gpu, logname):  # run phases one after another on one GPU; retry a failed phase once
    for args, tmo in steps:
        if wait_all([launch(args, gpu, tmo, logname)])[0] != 0:
            print("retrying once:", args[:4], flush=True); wait_all([launch(args, gpu, tmo, logname)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="driver"); ap.add_argument("--arm", default="ORM1"); ap.add_argument("--arms", default="BASE")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--vllm", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--bs", type=int, default=2); ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--ng", type=int, default=8); ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--vllm_util", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=10); ap.add_argument("--shuf", type=int, default=34); ap.add_argument("--max_train_min", type=float, default=150)
    ap.add_argument("--n_test", type=int, default=1000); ap.add_argument("--test_off", type=int, default=0)
    ap.add_argument("--n_bbh", type=int, default=250); ap.add_argument("--eval_max_new", type=int, default=2048)
    ap.add_argument("--n_samp", type=int, default=500); ap.add_argument("--k", type=int, default=8); ap.add_argument("--samp_seed", type=int, default=5151)
    ap.add_argument("--label", default=""); ap.add_argument("--tag", default=""); ap.add_argument("--out", default=HERE)
    ap.add_argument("--cpu_smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(WORK, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    if a.mode in ("train", "eval", "sample", "rm"):
        rc = 0
        try: {"train": do_train, "eval": do_eval, "sample": do_sample, "rm": do_rm}[a.mode](a)
        except BaseException:
            import traceback; traceback.print_exc(); rc = 1
        sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
    M3B = ["--model", "Qwen/Qwen2.5-3B-Instruct", "--label", "3B"]
    if a.cpu_smoke:
        sm = ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--vllm", "0", "--tag", "smoke_", "--out", "/tmp/m3orm_smoke"]
        tiny = ["--n_test", "2", "--n_bbh", "1", "--eval_max_new", "16"]
        print("train", wait_all([launch(["--mode", "train", "--arm", "ORM1"] + RS["ORM1"] + ["--steps", "1", "--bs", "4", "--ga", "1", "--ng", "2", "--max_new", "24"] + sm, None, 1200, "/tmp/smk_t.log")]))
        print("eval", wait_all([launch(["--mode", "eval", "--arms", "BASE,ORM1,ORM2"] + tiny + sm, None, 900, "/tmp/smk_e.log")]))
        print("sample", wait_all([launch(["--mode", "sample", "--arm", "ORM1", "--n_samp", "2", "--k", "3", "--eval_max_new", "16"] + sm, None, 900, "/tmp/smk_s.log")]))
        print("rm", wait_all([launch(["--mode", "rm", "--arm", "ORM1"] + sm, None, 900, "/tmp/smk_r.log")]))
        return
    try: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl[vllm]==1.18.1", "peft", "datasets"], timeout=1500, check=True)
    except Exception as e: print("pip failed:", repr(e)[:300], flush=True)
    os.system("nvidia-smi -L; pip list 2>/dev/null | grep -iE '^(torch|vllm|trl|transformers|peft) '")
    ngpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
    L = lambda n: f"/tmp/log_{n}.txt"
    # ---- in-job smoke: 1 GRPO step per replicate (with the RM reward), then every eval phase on a tiny slice
    sm = ["--tag", "smoke_", "--steps", "1", "--ga", "4", "--out", "/tmp/m3orm_smoke"]  # bs2*ga4 = 8 = one prompt group
    par = ngpu >= 2
    rcs = wait_all([launch(["--mode", "train", "--arm", "ORM1"] + RS["ORM1"] + sm, 0, 1200, "/tmp/sm_1.log"),
                    launch(["--mode", "train", "--arm", "ORM2"] + RS["ORM2"] + sm, 1 if par else 0, 1200, "/tmp/sm_2.log")] if par else
                   [launch(["--mode", "train", "--arm", "ORM1"] + RS["ORM1"] + sm, 0, 1200, "/tmp/sm_1.log")])
    if par and rcs[0] == 0 and rcs[1] != 0:
        print("parallel training failed on GPU1; going sequential", flush=True); os.system("tail -30 /tmp/sm_2.log"); par = False
    if rcs[0] != 0:
        os.system("tail -60 /tmp/sm_1.log"); print("SMOKE TRAIN FAILED; stopping"); sys.exit(1)
    tiny = ["--n_test", "4", "--n_bbh", "2", "--n_samp", "2", "--k", "3", "--eval_max_new", "128"]
    for args in (["--mode", "eval", "--arms", "BASE,ORM1"], ["--mode", "eval", "--arms", "BASE"] + M3B,
                 ["--mode", "sample", "--arm", "ORM1"], ["--mode", "rm", "--arm", "ORM1"]):
        if wait_all([launch(args + tiny + sm, 0, 1200, "/tmp/sm_e.log")])[0] != 0:
            os.system("tail -60 /tmp/sm_e.log"); print("SMOKE FAILED", args); sys.exit(1)
    print(f"SMOKE OK parallel={par}", flush=True)
    T = 160 * 60
    def oom(arm): return "OutOfMemoryError" in open(L(f"train_{arm}")).read()
    def retry(arms_gpus, rcs):  # on CUDA OOM, rerun that arm once with bs 1 x ga 32 (same 32 completions/step)
        bad = [(arm, g) for (arm, g), rc in zip(arms_gpus, rcs) if rc != 0 and oom(arm)]
        if bad:
            print("OOM retry with bs=1, vllm_util 0.27:", bad, flush=True)
            wait_all([launch(["--mode", "train", "--arm", arm, "--bs", "1", "--ga", "32", "--vllm_util", "0.27"] + RS[arm], g, T, L(f"train_{arm}")) for arm, g in bad])
    c0 = [(["--mode", "eval", "--arms", "BASE,ORM1"], 3600), (["--mode", "sample", "--arm", "ORM1"], 4500), (["--mode", "rm", "--arm", "ORM1"], 1800)]
    c1 = [(["--mode", "eval", "--arms", "ORM2"], 3000), (["--mode", "eval", "--arms", "BASE", "--n_bbh", "0"] + M3B, 2400),
          (["--mode", "sample", "--arm", "BASE"], 4500), (["--mode", "rm", "--arm", "BASE"], 1800)]
    if par:
        ag = [("ORM1", 0), ("ORM2", 1)]
        retry(ag, wait_all([launch(["--mode", "train", "--arm", arm] + RS[arm], g, T, L(f"train_{arm}")) for arm, g in ag]))
        import threading
        th = [threading.Thread(target=chain, args=(c, g, L(f"chain{g}"))) for c, g in ((c0, 0), (c1, 1))]
        for t in th: t.start()
        for t in th: t.join()
    else:  # one GPU: ORM1 + its evals first, so a partial run still gives one paired result
        retry([("ORM1", 0)], wait_all([launch(["--mode", "train", "--arm", "ORM1"] + RS["ORM1"], 0, T, L("train_ORM1"))]))
        chain(c0, 0, L("chain0"))
        retry([("ORM2", 0)], wait_all([launch(["--mode", "train", "--arm", "ORM2"] + RS["ORM2"], 0, T, L("train_ORM2"))]))
        chain(c1, 0, L("chain1"))
    for n in os.listdir("/tmp"):  # keep small filtered logs; gzip big per-item files
        if n.startswith("log_"): os.system(f"grep -vE 'it/s]|s/it]|Loading weights' /tmp/{n} | tail -400 > {a.out}/{n}")
    os.system(f"gzip -f {a.out}/eval_*.jsonl")
    print("DONE")

if __name__ == "__main__":
    main()
