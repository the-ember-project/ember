"""m3-grpo pilot (exploratory): can TRL GRPO + LoRA train Qwen2.5-1.5B-Instruct on MATH-train on a Kaggle T4 fast enough?
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

def run_one(a):
    import torch, dataclasses
    from datasets import Dataset
    from peft import LoraConfig
    import importlib  # Kaggle ships torchao 0.10 < peft's minimum 0.16 -> peft raises; tell peft torchao is absent
    for mod in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
        try: setattr(importlib.import_module(mod), "is_torchao_available", lambda: False)
        except Exception as e: print("torchao patch skipped", mod, repr(e)[:120], flush=True)
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    df = load_math("train")
    idx = list(range(len(df))); random.Random(31).shuffle(idx); idx = idx[:a.n_prompts]
    ds = Dataset.from_list([dict(prompt=[{"role": "system", "content": SYS_COT}, {"role": "user", "content": df.problem[i]}],
                                 gold=float(df.gold[i])) for i in idx])
    cuda = torch.cuda.is_available()
    want = dict(output_dir=f"/tmp/grpo_{a.tag}", learning_rate=a.lr, per_device_train_batch_size=a.bs,
                gradient_accumulation_steps=a.ga, num_generations=a.ng, max_completion_length=a.max_new, max_steps=a.steps,
                logging_steps=1, temperature=1.0, beta=0.0, use_vllm=a.vllm, vllm_mode="colocate",
                vllm_gpu_memory_utilization=0.35, vllm_max_model_length=1536, fp16=cuda, bf16=False,
                gradient_checkpointing=True, report_to="none", save_strategy="no", seed=a.seed,
                model_init_kwargs={"torch_dtype": torch.float16 if cuda else torch.float32}, use_cpu=not cuda)
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in want if k not in fields]
    if dropped: print("GRPOConfig: dropped unknown args", dropped, flush=True)
    cfg = GRPOConfig(**{k: v for k, v in want.items() if k in fields})
    times = []
    class T(TrainerCallback):
        def on_step_begin(self, *x, **k): self.t = time.time()
        def on_step_end(self, *x, **k): times.append(time.time() - self.t); print(f"step {len(times)} {times[-1]:.1f}s", flush=True)
    tr = GRPOTrainer(model=a.model, reward_funcs=[make_reward(a.reward)], args=cfg, train_dataset=ds,
                     peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                     callbacks=[T()])
    t0 = time.time(); tr.train()
    out = dict(tag=a.tag, vllm=a.vllm, steps=len(times), sec_per_step=times, total_s=round(time.time() - t0),
               stats=STATS, log=tr.state.log_history,
               gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None)
    json.dump(out, open(os.path.join(a.out, f"pilot_{a.tag}.json"), "w"), indent=1, default=str)
    print("RESULT", a.tag, "median s/step", sorted(times)[len(times) // 2] if times else None, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("--reward", default="GOLD")
    ap.add_argument("--vllm", type=int, default=1); ap.add_argument("--tag", default="vllm")
    ap.add_argument("--steps", type=int, default=12); ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--ga", type=int, default=4); ap.add_argument("--ng", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", default=HERE)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.one:
        rc = 0
        try: run_one(a)
        except BaseException:
            import traceback; traceback.print_exc(); rc = 1
        sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
    me = [sys.executable, os.path.abspath(__file__), "--one", "--out", a.out]
    if a.smoke:  # CPU: tiny model, 1 step, both reward kinds
        for rw in ("GOLD", "MAJ", "RANDOM"):
            r = subprocess.run(me + ["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--vllm", "0", "--tag", f"smoke_{rw}", "--reward", rw,
                                     "--steps", "1", "--n_prompts", "4", "--bs", "4", "--ga", "1", "--ng", "2", "--max_new", "24"], timeout=900)
            print("smoke", rw, "exit", r.returncode, flush=True)
        return
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "trl[vllm]==1.18.1", "peft", "datasets"], timeout=1500, check=True)
    except Exception as e: print("pip failed:", repr(e)[:300], flush=True)
    subprocess.run([sys.executable, "-m", "pip", "list"], stdout=open("/tmp/piplist.txt", "w"))
    os.system("grep -iE '^(torch|vllm|trl|transformers|peft) ' /tmp/piplist.txt")
    for tag, vl, steps, tmo in (("vllm", "1", a.steps, 1800), ("hf", "0", 2, 900)):
        t0 = time.time()
        p = subprocess.Popen(me + ["--vllm", vl, "--tag", tag, "--steps", str(steps)], start_new_session=True)
        try: rc = p.wait(timeout=tmo)
        except subprocess.TimeoutExpired:  # kill the whole group (vLLM engine children too) so the GPU is freed
            import signal; os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = "timeout"
        print(f"CONFIG {tag} exit={rc} {time.time()-t0:.0f}s", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
