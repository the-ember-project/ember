"""m3-dpo-iter driver: two chains (A, B), one per T4. Each chain: round-1 MIXR-DPO on train q 0..N-1 -> eval test
(BASE, R1) -> merge R1 -> round-2 MIXR-DPO from R1m on train q N..2N-1 -> eval test (R1m, R2).
Every chain first runs the whole sequence on a tiny slice (smoke) before the full run.
`--cpu_smoke`: logic check on CPU with 0.5B, sequential."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
CHAINS = [("A", 7071, 7072, 3041), ("B", 8081, 8082, 3042)]  # (tag, gen seed r1, gen seed r2, train seed)

def steps(tag, s1, s2, ts, n, tiny, extra):
    w = f"/tmp/iter_{tag}{'_smoke' if tiny else ''}"; out = os.path.join(HERE, tag) + ("_smoke_out" if tiny else "")
    if tiny: out = f"/tmp/iter_{tag}_smoke_out"
    r1m = w + "_r1m"; base_model = extra["model"]
    c = ["--backend", extra["backend"], "--n_train", str(n), "--train_seed", str(ts), "--out", out] + extra["more"]
    if tiny: c += ["--n_test", "4", "--cap", "4"]
    T = ["--split", "test"]
    return out, [
        ("gen1", ["--phase", "gen", "--model", base_model, "--seed", str(s1), "--q_off", "0", "--work", w + "_1"] + c),
        ("train1", ["--phase", "train", "--model", base_model, "--work", w + "_1"] + c),
        ("eval1", ["--phase", "eval", "--model", base_model, "--seed", str(s1), "--work", w + "_1", "--tag", "test_r1_"] + T + c),
        ("merge", ["--phase", "merge", "--model", base_model, "--work", w + "_1", "--merged", r1m] + c),
        ("gen2", ["--phase", "gen", "--model", r1m, "--seed", str(s2), "--q_off", str(n), "--work", w + "_2"] + c),
        ("train2", ["--phase", "train", "--model", r1m, "--q_off", str(n), "--work", w + "_2"] + c),
        ("eval2", ["--phase", "eval", "--model", r1m, "--seed", str(s2), "--work", w + "_2", "--tag", "test_r2_"] + T + c),
    ]

def chain(tag, s1, s2, ts, extra, n):
    for tiny in ((False,) if extra["backend"] == "hf" and n <= 3 else (True, False)):
        out, st = steps(tag, s1, s2, ts, 16 if tiny else n, tiny, extra)
        os.makedirs(out, exist_ok=True)
        for name, args in st:
            t0 = time.time(); rc = None
            for attempt in range(2):  # one retry (transient HF-hub errors); train is deterministic-ish, retry too
                try:
                    rc = subprocess.run([sys.executable, os.path.join(HERE, "run.py")] + args,
                                        timeout=(1200 if tiny else 6000)).returncode
                except subprocess.TimeoutExpired:
                    rc = -9
                print(f"[{tag}] {'SMOKE ' if tiny else ''}{name} attempt {attempt} rc={rc} {time.time()-t0:.0f}s", flush=True)
                if rc == 0: break
                time.sleep(20)
            if rc != 0:
                print(f"[{tag}] stopping chain at {name}", flush=True); return 1
    return 0

def main():
    if "--chain" in sys.argv:
        i = sys.argv.index("--chain"); tag = sys.argv[i + 1]; cpu = "--cpu" in sys.argv
        c = [x for x in CHAINS if x[0] == tag][0]
        if cpu:
            extra = dict(backend="hf", model="Qwen/Qwen2.5-0.5B-Instruct",
                         more=["--k_cot", "2", "--k_pal", "1", "--max_new", "48", "--max_len", "256", "--n_test", "2", "--cap", "3"])
            n = 3
        else:
            extra = dict(backend=sys.argv[sys.argv.index("--backend") + 1], model="Qwen/Qwen2.5-1.5B-Instruct", more=[])
            n = 2000 if extra["backend"] == "vllm" else 600
        sys.stdout.flush(); os._exit(chain(tag, *c[1:], extra, n))
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
    procs = []
    for k, (tag, *_r) in enumerate(CHAINS):
        os.makedirs(os.path.join(HERE, tag), exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--chain", tag, "--backend", backend] + (["--cpu"] if cpu else [])
        env = dict(os.environ)
        if ngpu >= 2: env["CUDA_VISIBLE_DEVICES"] = str(k)
        log = open(os.path.join(HERE, tag, "log.txt"), "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(f"started {tag} pid={p.pid} gpu={env.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        if ngpu < 2: p.wait(); print(f"{tag} exit={p.returncode}", flush=True)
        else: procs.append((tag, p)); time.sleep(60)
    for tag, p in procs:
        p.wait(); print(f"{tag} exit={p.returncode}", flush=True)
    for tag, *_r in CHAINS:
        try:
            fn = os.path.join(HERE, tag, "log.txt")
            keep = [l for l in open(fn, errors="replace") if "Processed prompts" not in l and "Capturing CUDA" not in l
                    and "it/s]" not in l and "s/it]" not in l][-500:]
            open(fn, "w").writelines(keep)
            print(f"--- {tag} log tail ---\n" + "".join(keep[-25:]), flush=True)
        except Exception as e: print(tag, "no log", e)
    os._exit(0)

if __name__ == "__main__":
    main()
