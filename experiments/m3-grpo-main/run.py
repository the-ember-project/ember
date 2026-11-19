"""m3-grpo-main: GRPO + LoRA on Qwen2.5-1.5B-Instruct (MATH-train) with GOLD vs MAJ (TTRL majority) vs RANDOM reward,
then greedy 2048-token eval on fresh MATH-test positions 2700-3699 (seed-2029 shuffle) + BBH-date drift. See README.
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

STATS = []
def make_reward(kind, seed=0):
    rng = random.Random(seed)
    def reward(prompts, completions, gold, **kw):
        preds = [num(boxed(text_of(c))) for c in completions]
        groups = {}
        for i, p in enumerate(prompts): groups.setdefault(json.dumps(p) if not isinstance(p, str) else p, []).append(i)
        maj = [None] * len(preds); majok = []
        for ids in groups.values():
            c = Counter(key(preds[i]) for i in ids if preds[i] is not None)
            m = c.most_common(1)[0][0] if c else None
            for i in ids: maj[i] = m
            majok.append(same(m, gold[ids[0]]))
        gold_r = [1.0 if same(p, g) else 0.0 for p, g in zip(preds, gold)]
        maj_r = [1.0 if same(p, m) else 0.0 for p, m in zip(preds, maj)]
        STATS.append(dict(gold_acc=sum(gold_r) / len(gold_r), boxed=sum(p is not None for p in preds) / len(preds),
                          maj_precision=sum(majok) / len(majok),
                          maj_reward_acc=sum(x == y for x, y in zip(gold_r, maj_r)) / len(preds)))
        if kind == "GOLD": return gold_r
        if kind == "MAJ": return maj_r
        if kind == "RANDOM": return [float(rng.random() < 0.5) for _ in preds]
        raise ValueError(kind)
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
    idx = list(range(len(df))); random.Random(31).shuffle(idx); idx = idx[:a.steps * 4 + 8]
    ds = Dataset.from_list([dict(prompt=[{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}],
                                 gold=float(df.gold[i])) for i in idx])
    cuda = torch.cuda.is_available()
    want = dict(output_dir=f"{WORK}/ckpt_{a.arm}{a.tag}", learning_rate=a.lr, lr_scheduler_type="constant", warmup_steps=0,
                per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.ga, num_generations=a.ng,
                max_completion_length=a.max_new, max_steps=a.steps, logging_steps=1, temperature=1.0, beta=0.0,
                use_vllm=bool(a.vllm), vllm_mode="colocate", vllm_gpu_memory_utilization=0.35, vllm_max_model_length=1536,
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

def do_eval(a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    df, idx = eval_items(a.n_test, a.test_off)
    bb = pd.read_parquet(hf_hub_download("lukaemon/bbh", "date_understanding/test-00000-of-00001.parquet", repo_type="dataset")).iloc[:a.n_bbh]
    tok = AutoTokenizer.from_pretrained(a.model)
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompts = [chat([{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}]) for i in idx]
    bprompts = [chat([{"role": "user", "content": q + "\n\n" + BBH_COT}]) for q in bb["input"]]
    arms = a.arms.split(",")
    if a.vllm:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(a.model, dtype="half", gpu_memory_utilization=0.85, max_model_len=a.eval_max_new + 640, seed=0,
                  enable_lora=True, max_lora_rank=16, max_loras=1)
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=a.eval_max_new)
        def gen(ps, arm, j):
            kw = {} if arm == "BASE" else dict(lora_request=LoRARequest(f"a{j}", j, f"{WORK}/lora_{arm}{a.tag}"))
            return [o.outputs[0].text for o in llm.generate(ps, sp, **kw)]
    else:  # CPU smoke path only
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        def gen(ps, arm, j):
            m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
            if arm != "BASE": m = PeftModel.from_pretrained(m, f"{WORK}/lora_{arm}{a.tag}").merge_and_unload()
            res = []
            for p in ps:
                enc = tok(p, return_tensors="pt")
                o = m.generate(**enc, max_new_tokens=a.eval_max_new, do_sample=False)
                res.append(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
            return res
    for j, arm in enumerate(arms):
        t0 = time.time()
        outs = gen(prompts, arm, j + 1); bouts = gen(bprompts, arm, j + 1) if len(bprompts) else []
        with open(os.path.join(a.out, f"{a.tag}eval_{arm}.jsonl"), "w") as f:
            for i, o in zip(idx, outs):
                p = extract(o)
                f.write(json.dumps(dict(idx=i, subject=df.subject[i], level=df.level[i], gold=float(df.gold[i]), pred=p,
                                        correct=same(p, float(df.gold[i])), len=len(o), boxed=boxed(o) is not None, tail=o[-300:])) + "\n")
            for i, o in enumerate(bouts):
                m = re.findall(r"answer is:?\s*\**\s*\(?([A-G])\)?(?![a-zA-Z])", o)
                p = f"({m[-1]})" if m else None
                f.write(json.dumps(dict(bbh_idx=i, pred=p, correct=p == bb["target"].iloc[i], tail=o[-300:])) + "\n")
        acc = sum(same(extract(o), float(df.gold[i])) for i, o in zip(idx, outs)) / len(idx)
        print(f"EVAL {arm}{a.tag}: math={acc:.4f} boxed={sum(boxed(o) is not None for o in outs)/len(outs):.3f} {time.time()-t0:.0f}s", flush=True)

def launch(args, gpu, tmo, log):
    env = dict(os.environ)
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["MASTER_PORT"] = str(29500 + 17 * (gpu + 1))
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + args, start_new_session=True, env=env,
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    p._tmo, p._t0, p._log, p._args = tmo, time.time(), log, args
    return p

def wait_all(ps):
    rcs = []
    for p in ps:
        try: rc = p.wait(timeout=max(1, p._tmo - (time.time() - p._t0)))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = "timeout"
        os.system(f"grep -E 'EVAL|TRAINED|Error|error|Traceback|time guard' {p._log} | tail -8; tail -3 {p._log}")
        print(f"PROC {' '.join(p._args[:4])} exit={rc} {time.time()-p._t0:.0f}s", flush=True); rcs.append(rc)
    return rcs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="driver"); ap.add_argument("--arm", default="GOLD"); ap.add_argument("--arms", default="BASE")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--vllm", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--ga", type=int, default=4)
    ap.add_argument("--ng", type=int, default=8); ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--max_train_min", type=float, default=150)
    ap.add_argument("--n_test", type=int, default=1000); ap.add_argument("--test_off", type=int, default=2700)
    ap.add_argument("--n_bbh", type=int, default=250); ap.add_argument("--eval_max_new", type=int, default=2048)
    ap.add_argument("--tag", default=""); ap.add_argument("--out", default=HERE); ap.add_argument("--cpu_smoke", action="store_true")
    a = ap.parse_args(); os.makedirs(WORK, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    if a.mode in ("train", "eval"):
        rc = 0
        try: (do_train if a.mode == "train" else do_eval)(a)
        except BaseException:
            import traceback; traceback.print_exc(); rc = 1
        sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
    if a.cpu_smoke:
        sm = ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--vllm", "0", "--tag", "smoke_", "--out", "/tmp/m3grpo_smoke"]
        for arm in ("GOLD", "MAJ", "RANDOM"):
            print(arm, wait_all([launch(["--mode", "train", "--arm", arm, "--steps", "1", "--bs", "4", "--ga", "1", "--ng", "2", "--max_new", "24"] + sm, None, 900, f"/tmp/smk_{arm}.log")]))
        print("eval", wait_all([launch(["--mode", "eval", "--arms", "BASE,GOLD,MAJ,RANDOM", "--n_test", "2", "--n_bbh", "1", "--eval_max_new", "16"] + sm, None, 900, "/tmp/smk_eval.log")]))
        return
    try: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl[vllm]==1.18.1", "peft", "datasets"], timeout=1500, check=True)
    except Exception as e: print("pip failed:", repr(e)[:300], flush=True)
    os.system("nvidia-smi -L; pip list 2>/dev/null | grep -iE '^(torch|vllm|trl|transformers|peft) '")
    ngpu = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
    L = lambda n: f"/tmp/log_{n}.txt"
    # ---- in-job smoke: 1 GRPO step per arm (two in parallel if 2 GPUs), then a 4-item eval of BASE + both adapters
    sm = ["--tag", "smoke_", "--steps", "1", "--ga", "1", "--out", "/tmp/m3grpo_smoke"]
    par = ngpu >= 2
    rcs = wait_all([launch(["--mode", "train", "--arm", "GOLD"] + sm, 0, 1200, "/tmp/sm_g.log"),
                    launch(["--mode", "train", "--arm", "MAJ"] + sm, 1 if par else 0, 1200, "/tmp/sm_m.log")] if par else
                   [launch(["--mode", "train", "--arm", "GOLD"] + sm, 0, 1200, "/tmp/sm_g.log")])
    if par and rcs[0] == 0 and rcs[1] != 0:
        print("parallel training failed on GPU1; going sequential", flush=True); os.system("cat /tmp/sm_m.log | tail -30"); par = False
    if rcs[0] != 0:
        os.system("tail -60 /tmp/sm_g.log"); print("SMOKE TRAIN FAILED; stopping"); sys.exit(1)
    rcs = wait_all([launch(["--mode", "eval", "--arms", "BASE,GOLD", "--n_test", "4", "--n_bbh", "2"] + sm, 0, 1200, "/tmp/sm_e.log")])
    if rcs[0] != 0: os.system("tail -60 /tmp/sm_e.log"); print("SMOKE EVAL FAILED; stopping"); sys.exit(1)
    print(f"SMOKE OK parallel={par}", flush=True)
    T, E = 170 * 60, 75 * 60
    if par:
        wait_all([launch(["--mode", "train", "--arm", "GOLD"], 0, T, L("train_GOLD")), launch(["--mode", "train", "--arm", "MAJ"], 1, T, L("train_MAJ"))])
        wait_all([launch(["--mode", "train", "--arm", "RANDOM"], 0, T, L("train_RANDOM")),
                  launch(["--mode", "eval", "--arms", "BASE,GOLD,MAJ"], 1, E, L("eval_1"))])
        wait_all([launch(["--mode", "eval", "--arms", "RANDOM"], 0, E, L("eval_2"))])
    else:  # deciding arm first
        for arm, ev in (("GOLD", "BASE,GOLD"), ("MAJ", "MAJ"), ("RANDOM", "RANDOM")):
            wait_all([launch(["--mode", "train", "--arm", arm], 0, T, L(f"train_{arm}"))])
            wait_all([launch(["--mode", "eval", "--arms", ev], 0, E, L(f"eval_{arm}"))])
    for n in os.listdir("/tmp"):  # keep small filtered logs; gzip big per-item files
        if n.startswith("log_"): os.system(f"grep -vE 'it/s]|s/it]|Loading weights' /tmp/{n} | tail -400 > {a.out}/{n}")
    os.system(f"gzip -f {a.out}/eval_*.jsonl")
    print("DONE")

if __name__ == "__main__":
    main()
