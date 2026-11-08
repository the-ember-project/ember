"""Generative state-tracking task (gridwalk): an agent starts at (0,0) facing north and follows a list of
moves (forward N, turn left/right, turn around). The model must output the final (x, y) position.
Free-form answer (not multiple choice), exact checker. Modes: cot (reason step by step) or pal (write a
Python simulation). Resumable jsonl, all raw texts saved.
Usage: python run.py --mode cot|pal --n 50 --offset 0 --k 1 --temp 0 --out f.jsonl [--steps 10]"""
import argparse, json, os, random, re, subprocess, sys, tempfile, time

DIRS = ["north", "east", "south", "west"]
DV = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}

def make_task(i, steps, seed=777):
    r = random.Random(seed * 100003 + i)
    acts, x, y, d = [], 0, 0, 0
    for _ in range(steps):
        u = r.random()
        if u < 0.55:
            n = r.randint(1, 4); acts.append(f"walk forward {n} step{'s' if n > 1 else ''}")
            dx, dy = DV[DIRS[d]]; x += dx * n; y += dy * n
        elif u < 0.77: acts.append("turn left"); d = (d - 1) % 4
        elif u < 0.97: acts.append("turn right"); d = (d + 1) % 4
        else: acts.append("turn around"); d = (d + 2) % 4
    q = ("A robot stands on an infinite grid at position (0, 0), facing north. North increases y, east "
         "increases x. Turning does not change its position. It follows these instructions in order:\n"
         + "\n".join(f"{k+1}. {a}" for k, a in enumerate(acts)) + "\nWhat is the robot's final position (x, y)?")
    return q, (x, y)

SYS_COT = ("You are a careful reasoner. Track the robot's position and facing direction after every "
           "instruction, one line per instruction. At the very end write the answer on its own line as "
           "'Final answer: (x, y)'.")
SYS_PAL = ("You are a careful programmer. Write a short Python program that simulates the robot instruction "
           "by instruction and prints the final position as a tuple, e.g. print((x, y)). Output only the code "
           "in one ```python block.")

PAIR = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"

def parse_cot(text):
    m = re.findall(r"[Ff]inal answer[^\n(]*" + PAIR, text)
    if m: return f"{int(m[-1][0])},{int(m[-1][1])}"
    m = re.findall(PAIR, text)
    return f"{int(m[-1][0])},{int(m[-1][1])}" if m else None

def run_code(text, timeout=10):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[0] if m else text
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        out, err = p.stdout.strip(), p.stderr.strip()[-200:]
    except subprocess.TimeoutExpired:
        out, err = "", "TIMEOUT"
    os.unlink(path)
    m = re.findall(PAIR, out)
    if not m:
        nums = re.findall(r"-?\d+", out)
        m = [nums[-2:]] if len(nums) >= 2 else []
    return code, out[-200:], err, (f"{int(m[-1][0])},{int(m[-1][1])}" if m else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    ap.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--mode", choices=["cot", "pal"], required=True)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--task_seed", type=int, default=777)
    ap.add_argument("--k", type=int, default=1); ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--sample_seed", type=int, default=0); ap.add_argument("--max_tokens", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    llm = Llama(model_path=hf_hub_download(a.repo, a.file), n_ctx=2048, n_threads=os.cpu_count(), verbose=False, seed=0)
    done = {json.loads(l)["idx"] for l in open(a.out)} if os.path.exists(a.out) else set()
    c = tok = n = 0; t0 = time.time()
    with open(a.out, "a") as f:
        for i in range(a.offset, a.offset + a.n):
            if i in done: continue
            q, gold = make_task(i, a.steps, a.task_seed); gold = f"{gold[0]},{gold[1]}"
            recs = []
            for j in range(a.k):
                r = llm.create_chat_completion(messages=[{"role": "system", "content": SYS_COT if a.mode == "cot" else SYS_PAL},
                                                         {"role": "user", "content": q}],
                                               temperature=a.temp, max_tokens=a.max_tokens, seed=a.sample_seed * 1000 + j)
                text = r["choices"][0]["message"]["content"]; tk = r["usage"]["completion_tokens"]
                if a.mode == "cot": rec = {"text": text, "pred": parse_cot(text), "tokens": tk}
                else:
                    code, out, err, p_ = run_code(text)
                    rec = {"text": text, "out": out, "err": err, "pred": p_, "tokens": tk}
                recs.append(rec); tok += tk
            preds = [x["pred"] for x in recs]
            ok = preds[0] == gold; c += ok; n += 1
            f.write(json.dumps({"idx": i, "gold": gold, "preds": preds, "ok": ok, "samples": recs}) + "\n"); f.flush()
    dt = time.time() - t0
    print(f"SUMMARY mode={a.mode} n={n} acc(first)={c/max(n,1):.3f} tok/s={tok/max(dt,1e-9):.1f} s/q={dt/max(n,1):.1f}")

if __name__ == "__main__":
    main()
