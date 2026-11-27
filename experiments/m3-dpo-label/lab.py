"""m3-dpo-label driver. Same 3000 GSM8K-train questions for all arms.
Stage 1 (parallel): GPU0 = 1.5B gen (greedy+7 CoT, greedy+4 PAL); GPU1 = 3B teacher gen (greedy+4 CoT, greedy+1 PAL).
Stage 2 (parallel): GPU0 trains MIXR, T3D (train seed 4051); GPU1 trains GOLD, T3D (train seed 4052). Equal pair count E.
Stage 3 (parallel): GPU0 evals BASE, MIXR, T3D -> G0/; GPU1 evals GOLD, T3D -> G1/ (greedy CoT, full GSM8K test).
Stage 4 (#48): same split, mix7 vote (2 CoT + 5 PAL, T=0.7) per arm -> test_mix_<arm>.jsonl.gz (M5 check).
The whole pipeline first runs on a tiny slice (smoke). `--cpu_smoke`: logic check on CPU with 0.5B/0.5B, sequential."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run.py")

def stages(tiny, cpu, backend):
    sfx = "_smoke" if tiny else ""
    w15, w3, t0, t1 = [f"/tmp/lab_{x}{sfx}" for x in ("15", "3b", "tr0", "tr1")]
    o0 = f"/tmp/lab_G0{sfx}_out" if tiny else os.path.join(HERE, "G0")
    o1 = f"/tmp/lab_G1{sfx}_out" if tiny else os.path.join(HERE, "G1")
    small, big = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct") if cpu else ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
    n = 3 if cpu else (16 if tiny else (3000 if backend == "vllm" else 600))
    c = ["--backend", backend, "--n_train", str(n), "--teacher_work", w3]
    if cpu: c += ["--max_new", "48", "--max_len", "256", "--n_test", "2", "--cap", "3"]
    elif tiny: c += ["--n_test", "4", "--cap", "4"]
    T = ["--split", "test", "--tag", "test_"]
    return [
        [("gen15", ["--phase", "gen", "--model", small, "--seed", "9091", "--work", w15, "--out", o0, "--tag", "_15"] + c
          + (["--k_cot", "2", "--k_pal", "1"] if cpu else [])),
         ("gen3b", ["--phase", "gen", "--model", big, "--seed", "9092", "--work", w3, "--out", o1, "--tag", "_3b",
                    "--k_cot", "2" if cpu else "4", "--k_pal", "1"] + c)],
        [("trainA", ["--phase", "train", "--model", small, "--gen_work", w15, "--work", t0, "--out", o0, "--arms", "MIXR,T3D",
                     "--train_seed", "4051", "--tag", "_A"] + c),
         ("trainB", ["--phase", "train", "--model", small, "--gen_work", w15, "--work", t1, "--out", o1, "--arms", "GOLD,T3D",
                     "--train_seed", "4052", "--tag", "_B"] + c)],
        [("evalA", ["--phase", "eval", "--model", small, "--seed", "9093", "--work", t0, "--out", o0, "--arms", "MIXR,T3D",
                    "--with_base", "1"] + T + c),
         ("evalB", ["--phase", "eval", "--model", small, "--seed", "9094", "--work", t1, "--out", o1, "--arms", "GOLD,T3D"] + T + c)],
        # session #48: M5 stage -- mix7 vote per arm on the full test (BASE on GPU0)
        [("mixA", ["--phase", "mix", "--model", small, "--seed", "9095", "--work", t0, "--out", o0, "--arms", "T3D,MIXR",
                   "--with_base", "1"] + T + c),
         ("mixB", ["--phase", "mix", "--model", small, "--seed", "9096", "--work", t1, "--out", o1, "--arms", "T3D,GOLD"] + T + c)],
    ]

def run_step(name, args, gpu, tiny, ngpu):
    env = dict(os.environ)
    if ngpu >= 2: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(args[args.index("--out") + 1], exist_ok=True)
    rc = None; t0 = time.time()
    for attempt in range(2):  # one retry (transient HF-hub errors)
        try:
            with open(os.path.join(args[args.index("--out") + 1], f"log_{name}.txt"), "a") as lf:
                rc = subprocess.run([sys.executable, RUN] + args, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=(1500 if tiny else 7200)).returncode
        except subprocess.TimeoutExpired:
            rc = -9
        print(f"{'SMOKE ' if tiny else ''}{name} gpu{gpu} attempt {attempt} rc={rc} {time.time()-t0:.0f}s", flush=True)
        if rc == 0: break
        time.sleep(20)
    return rc

def pipeline(tiny, cpu, backend, ngpu):
    from concurrent.futures import ThreadPoolExecutor
    for st in stages(tiny, cpu, backend):
        if ngpu >= 2:
            with ThreadPoolExecutor(2) as ex:
                rcs = list(ex.map(lambda kv: run_step(kv[1][0], kv[1][1], kv[0], tiny, ngpu), enumerate(st)))
        else:
            rcs = [run_step(nm, ar, k, tiny, ngpu) for k, (nm, ar) in enumerate(st)]
        # gen3b / T3D failure is tolerated (MIXR vs GOLD still informative); a gen15 failure is not
        if rcs[0] != 0: print("stage failed on GPU0; stopping", flush=True); return 1
    return 0

def main():
    cpu = "--cpu_smoke" in sys.argv
    ngpu, backend = 0, "hf"
    if not cpu:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "peft"], timeout=1500, check=False)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
        try:
            import torch; ngpu = torch.cuda.device_count()
        except Exception: ngpu = 1
        try:
            subprocess.run([sys.executable, "-c", "from vllm import LLM, SamplingParams; "
                            "l=LLM('Qwen/Qwen2.5-1.5B-Instruct', dtype='half', max_model_len=1024, gpu_memory_utilization=0.5); "
                            "print('VLLM_OK', l.generate(['2+2='], SamplingParams(max_tokens=8))[0].outputs[0].text)"],
                           timeout=900, check=True)
            backend = "vllm"
        except Exception as e: print("vLLM unavailable, HF fallback:", repr(e)[:300], flush=True)
    print(f"ngpu={ngpu} backend={backend}", flush=True)
    rc = 0
    for tiny in ((False,) if cpu else (True, False)):
        rc = pipeline(tiny, cpu, backend, ngpu)
        if rc: print(f"pipeline {'smoke' if tiny else 'full'} failed", flush=True); break
    for d in ("G0", "G1"):  # shrink logs (drop progress bars), print tails into the job output
        dd = os.path.join(HERE, d)
        for fn in sorted(os.listdir(dd)) if os.path.isdir(dd) else []:
            if not fn.startswith("log_"): continue
            p = os.path.join(dd, fn)
            keep = [x for x in open(p, errors="replace") if "Processed prompts" not in x and "Capturing CUDA" not in x
                    and "it/s]" not in x and "s/it]" not in x][-400:]
            open(p, "w").writelines(keep); print(f"--- {d}/{fn} tail ---\n" + "".join(keep[-12:]), flush=True)
    print("DONE rc", rc, flush=True); sys.stdout.flush(); os._exit(rc)

if __name__ == "__main__":
    main()
