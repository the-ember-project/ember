"""BBH state-tracking/spatial tasks: greedy CoT, k CoT samples, greedy PAL, k PAL samples with HF transformers
(GPU on Kaggle; CPU works for smoke tests). Writes one jsonl per task with predictions only (+ greedy texts).
Resumable. Usage: python run.py [--model Qwen/Qwen2.5-1.5B-Instruct] [--n 250] [--k 5] [--tasks a,b]"""
import argparse, json, os, re, subprocess, sys, tempfile, time
import torch, pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

COT = ("Think step by step, keeping track of the full state after every step. "
       "End with a line of the form 'So the answer is X.' where X is {fmt}.")
PAL = ("Write a short Python program that simulates the situation step by step with explicit variables "
       "for the state, then prints only the final answer: {pfmt}. Output only the code in one ```python block.")
FMT = {"navigate": ("Yes or No", "print exactly Yes or No"),
       "mc": ("the option letter in parentheses, e.g. (A)", "print the text of the correct option (not the letter)")}

def options(q):  # [(letter, text)]
    return re.findall(r"^\(([A-G])\)\s*(.+?)\s*$", q, re.M)

def parse_cot(task, text):
    if task == "navigate":
        m = re.findall(r"\b(Yes|No)\b", text, re.I)
        return m[-1].capitalize() if m else None
    m = re.findall(r"answer is:?\s*\**\s*\(?([A-G])\)?(?![a-zA-Z])", text)  # fixed: parens optional
    if m: return f"({m[-1]})"
    m = re.findall(r"\(([A-G])\)", text)
    return f"({m[-1]})" if m else None

def run_code(text, timeout=10):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = m[0] if m else text
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        out = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        out = ""
    os.unlink(path); return out

def parse_pal(task, q, out):
    if not out: return None
    last = out.strip().splitlines()[-1].strip().strip(".'\"")
    if task == "navigate":
        l = last.lower()
        return "Yes" if l in ("yes", "true") else "No" if l in ("no", "false") else None
    opts = options(q)
    for L, t in opts:  # exact text match first
        if last.lower() == t.lower().strip("."): return f"({L})"
    hit = [L for L, t in opts if t.lower().strip(".") in last.lower()]
    if len(hit) == 1: return f"({hit[0]})"   # accept containment only when exactly one option text appears
    m = re.fullmatch(r"\(?([A-G])\)?", last)
    if m: return f"({m.group(1)})"
    w = last.lower().strip(".")  # fixed: output is part of exactly one option text ("black" -> "black ball")
    hit = [L for L, t in opts if w and re.search(r"\b" + re.escape(w) + r"\b", t.lower())]
    if len(hit) == 1: return f"({hit[0]})"
    m = None
    return f"({m.group(1)})" if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tasks", default="tracking_shuffled_objects_three_objects,logical_deduction_five_objects,date_understanding")
    ap.add_argument("--n", type=int, default=250); ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--max_new", type=int, default=512); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

    def gen(prompts, k, temp):
        enc = tok([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                   for p in prompts], return_tensors="pt", padding=True).to(dev)
        kw = dict(do_sample=True, temperature=temp, top_p=1.0, num_return_sequences=k) if temp > 0 else dict(do_sample=False)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id, **kw)
        o = o[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(o, skip_special_tokens=True)
        stop = {tok.pad_token_id, tok.eos_token_id}
        ntok = [sum(int(x) not in stop for x in r) for r in o]
        return [texts[i * k:(i + 1) * k] for i in range(len(prompts))], [ntok[i * k:(i + 1) * k] for i in range(len(prompts))]

    for task in a.tasks.split(","):
        d = pd.read_parquet(hf_hub_download("lukaemon/bbh", f"{task}/test-00000-of-00001.parquet", repo_type="dataset"))
        kind = "navigate" if task == "navigate" else "mc"; fmt, pfmt = FMT[kind]
        out = os.path.join(a.outdir, f"{task}.jsonl")
        done = {json.loads(l)["idx"] for l in open(out)} if os.path.exists(out) else set()
        todo = [i for i in range(min(a.n, len(d))) if i not in done]
        t0 = time.time(); right = {"cg": 0, "pg": 0}; cnt = 0
        for b in range(0, len(todo), a.bs):
            ids = todo[b:b + a.bs]; qs = [d.iloc[i]["input"] for i in ids]
            torch.manual_seed(a.seed * 100000 + ids[0])
            cp = [q + "\n\n" + COT.format(fmt=fmt) for q in qs]; pp = [q + "\n\n" + PAL.format(pfmt=pfmt) for q in qs]
            cg, cgt = gen(cp, 1, 0); cs, cst = gen(cp, a.k, a.temp); pg, pgt = gen(pp, 1, 0); ps, pst = gen(pp, a.k, a.temp)
            pal_outs = {}
            with open(out, "a") as f:
                for j, i in enumerate(ids):
                    q, gold = qs[j], d.iloc[i]["target"]
                    pg_out = run_code(pg[j][0]); pal_outs[j] = [run_code(t)[-80:] for t in ps[j]]
                    r = dict(idx=int(i), gold=gold,
                             cot_greedy=parse_cot(kind, cg[j][0]), cot_samples=[parse_cot(kind, t) for t in cs[j]],
                             pal_greedy=parse_pal(kind, q, pg_out), pal_samples=[parse_pal(kind, q, o) for o in pal_outs[j]],
                             tok=dict(cg=cgt[j][0], cs=sum(cst[j]), pg=pgt[j][0], ps=sum(pst[j])),
                             cot_greedy_text=cg[j][0][-400:], pal_greedy_code=pg[j][0][-400:], pal_greedy_out=pg_out[-100:],
                             cot_sample_tails=[t[-150:] for t in cs[j]], pal_sample_outs=pal_outs[j])
                    f.write(json.dumps(r) + "\n"); cnt += 1
                    right["cg"] += r["cot_greedy"] == gold; right["pg"] += r["pal_greedy"] == gold
            print(f"{task} {cnt}/{len(todo)} greedyCoT={right['cg']/cnt:.3f} greedyPAL={right['pg']/cnt:.3f} {time.time()-t0:.0f}s", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
