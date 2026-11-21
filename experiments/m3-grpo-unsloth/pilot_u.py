"""m3-grpo-unsloth pilot (exploratory, issue #8). Copy of m3-grpo/pilot.py helpers. Same GRPO config as the m3-grpo pilot;
GPU0 = Unsloth (FastLanguageModel, fast_inference vLLM), GPU1 = stock TRL+peft+vLLM colocate, in the same job/env.
Original m3-grpo pilot doc:: can TRL GRPO + LoRA train Qwen2.5-1.5B-Instruct on MATH-train on a Kaggle T4 fast enough?
Measures seconds/step (4 prompts x 8 completions per optimizer step), reward trend, completion lengths, truncation, and how
often the group-majority answer equals gold (TTRL label precision). Each config runs in a subprocess with a timeout and
hard-exits (vLLM shutdown-hang lesson, m3-stack run 1).
CPU smoke: python pilot.py --smoke
"""
import argparse, json, math, os, random, re, subprocess, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_COT = ("You are a careful math solver. Reason step by step, then give the final answer "
           "(a number: integer, decimal or fraction) in the form \\boxed{answer}.")  # identical to m3-math-stack
SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

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

def same(a, b): return a is not None and b is not None and abs(a - b) <= 1e-4 * max(1.0, abs(b))
def key(v): return None if v is None else round(v, 4)
def text_of(c): return c[-1]["content"] if isinstance(c, list) else c

def load_math(split):
    from huggingface_hub import hf_hub_download
    import pandas as pd
    dfs = []
    for sj in SUBJ:
        for t in range(3):  # transient hub errors killed m3-stack run 2's g3 phase
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
        STATS.append(dict(gold_acc=sum(gold_r) / len(gold_r), boxed=sum(p is not None for p in preds) / len(preds),
                          maj_precision=sum(majok) / len(majok)))
        if kind == "GOLD": return gold_r
        if kind == "MAJ": return [1.0 if same(p, m) else 0.0 for p, m in zip(preds, maj)]
        if kind == "RANDOM": return [float(rng.random() < 0.5) for _ in preds]
        raise ValueError(kind)
    reward.__name__ = f"r_{kind}"
    return reward

def make_ds(a):
    from datasets import Dataset
    df = load_math("train")
    idx = list(range(len(df))); random.Random(31).shuffle(idx); idx = idx[:a.n_prompts]
    return Dataset.from_list([dict(prompt=[{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}],
                                   gold=float(df.gold[i])) for i in idx])

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
def run_one(a):
    if a.stack == "unsloth":
        from unsloth import FastLanguageModel  # must be imported before trl/transformers so its patches apply
    import torch, dataclasses
    cuda = torch.cuda.is_available()
    if a.stack == "stock":
        import importlib  # Kaggle torchao 0.10 < peft min: tell peft torchao is absent
        for mod in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
            try: setattr(importlib.import_module(mod), "is_torchao_available", lambda: False)
            except Exception as e: print("torchao patch skipped", mod, repr(e)[:120], flush=True)
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    ds = make_ds(a)
    want = dict(output_dir=f"/tmp/grpo_{a.tag}", learning_rate=a.lr, per_device_train_batch_size=a.bs,
                gradient_accumulation_steps=a.ga, num_generations=a.ng, max_completion_length=a.max_new, max_steps=a.steps,
                logging_steps=1, temperature=1.0, beta=0.0, use_vllm=bool(a.vllm), fp16=cuda, bf16=False,
                report_to="none", save_strategy="no", seed=a.seed)
    if a.stack == "stock":
        want.update(vllm_mode="colocate", vllm_gpu_memory_utilization=0.35, vllm_max_model_length=1536, gradient_checkpointing=True,
                    model_init_kwargs={"torch_dtype": torch.float16 if cuda else torch.float32}, use_cpu=not cuda)
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in want if k not in fields]
    if dropped: print("GRPOConfig: dropped unknown args", dropped, flush=True)
    cfg = GRPOConfig(**{k: v for k, v in want.items() if k in fields})
    times = []
    class T(TrainerCallback):
        def on_step_begin(self, *x, **k): self.t = time.time()
        def on_step_end(self, *x, **k): times.append(time.time() - self.t); print(f"step {len(times)} {times[-1]:.1f}s", flush=True)
    t_load = time.time()
    if a.stack == "unsloth":
        model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=1536, load_in_4bit=False, dtype=None,
                                                       fast_inference=bool(a.vllm), max_lora_rank=16, gpu_memory_utilization=0.35)
        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                                 use_gradient_checkpointing="unsloth", random_state=a.seed)
        tr = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[make_reward(a.reward)], args=cfg, train_dataset=ds, callbacks=[T()])
    else:
        from peft import LoraConfig
        tr = GRPOTrainer(model=a.model, reward_funcs=[make_reward(a.reward)], args=cfg, train_dataset=ds, callbacks=[T()],
                         peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=TARGETS))
    load_s = round(time.time() - t_load)
    t0 = time.time(); tr.train()
    out = dict(tag=a.tag, stack=a.stack, vllm=a.vllm, load_s=load_s, steps=len(times), sec_per_step=times, total_s=round(time.time() - t0),
               stats=STATS, log=tr.state.log_history,
               gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None)
    json.dump(out, open(os.path.join(a.out, f"pilot_{a.tag}.json"), "w"), indent=1, default=str)
    print("RESULT", a.tag, "median s/step (steps 2+)", sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else None, flush=True)

def launch(cmd, gpu, tmo, log):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu); env["MASTER_PORT"] = str(29500 + gpu)
    p = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    return p, tmo, time.time(), log

def wait(job):
    p, tmo, t0, log = job
    try: rc = p.wait(timeout=max(1, tmo - (time.time() - t0)))
    except subprocess.TimeoutExpired:
        import signal; os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = "timeout"
    print(f"=== {log} exit={rc} {time.time()-t0:.0f}s", flush=True)
    os.system(f"grep -E 'RESULT|step [0-9]+ |Error|error|dropped|Unsloth' {log} | grep -v Warning | tail -40")
    return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true"); ap.add_argument("--stack", default="stock")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--reward", default="GOLD")
    ap.add_argument("--vllm", type=int, default=1); ap.add_argument("--tag", default="x")
    ap.add_argument("--steps", type=int, default=12); ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--ga", type=int, default=4); ap.add_argument("--ng", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", default=HERE); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.one:
        rc = 0
        try: run_one(a)
        except BaseException:
            import traceback; traceback.print_exc(); rc = 1
        sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
    me = [sys.executable, os.path.abspath(__file__), "--one", "--out", a.out]
    if not a.dry:
        t0 = time.time()
        try: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth", "vllm", "trl", "peft", "datasets"], timeout=1500, check=True)
        except Exception as e: print("pip failed:", repr(e)[:300], flush=True)
        print(f"INSTALL {time.time()-t0:.0f}s", flush=True)
        os.system("nvidia-smi -L; pip list 2>/dev/null | grep -iE '^(torch|vllm|trl|transformers|peft|unsloth|unsloth_zoo|torchao) '")
    # parallel: Unsloth on GPU0, stock on GPU1 (same env, same config, same prompts/seed)
    jobs = [launch(me + ["--stack", "unsloth", "--tag", "unsloth"] + (["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--vllm", "0", "--steps", "1", "--n_prompts", "4", "--bs", "4", "--ga", "1", "--ng", "2", "--max_new", "16"] if a.dry else []), 0, 1800, os.path.join(a.out, "log_unsloth.txt")),
            launch(me + ["--stack", "stock", "--tag", "stock"] + (["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--vllm", "0", "--steps", "1", "--n_prompts", "4", "--bs", "4", "--ga", "1", "--ng", "2", "--max_new", "16"] if a.dry else []), 1, 1800, os.path.join(a.out, "log_stock.txt"))]
    rcs = [wait(j) for j in jobs]
    if rcs[0] != 0 and not a.dry:  # fallback: Unsloth kernels with HF generation (no fast_inference), 3 steps
        wait(launch(me + ["--stack", "unsloth", "--tag", "unsloth_hf", "--vllm", "0", "--steps", "3"], 0, 900, os.path.join(a.out, "log_unsloth_hf.txt")))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
