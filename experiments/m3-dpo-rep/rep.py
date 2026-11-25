"""m3-dpo-rep driver: two fresh replicates of the m3-stack MIXR label-free DPO recipe, evaluated greedy on the full
GSM8K test set. One replicate per T4 (CUDA_VISIBLE_DEVICES), each a full run.py pipeline (smoke pass, gen, train,
test eval of BASE + MIXR). `--cpu_smoke` runs both sequentially on CPU with 0.5B and tiny sizes (logic check only)."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
REPS = [("R2", 5051, 3031), ("R3", 6061, 3032)]  # (tag, vLLM generation seed, training seed); m3-stack run 3 = R1 (4041, 2029)

def main():
    cpu = "--cpu_smoke" in sys.argv
    if cpu:
        extra = ["--backend", "hf", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--n_train", "3", "--n_test", "2",
                 "--k_cot", "2", "--k_pal", "1", "--max_new", "48", "--cap", "3", "--max_len", "256"]
        ngpu = 0
    else:
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
            extra = ["--backend", "vllm"]
        except Exception as e:
            print("vLLM unavailable, HF fallback:", repr(e)[:300], flush=True); extra = ["--backend", "hf"]
    print(f"ngpu={ngpu} extra={extra}", flush=True)
    procs = []
    for k, (tag, seed, tseed) in enumerate(REPS):
        out = os.path.join(HERE, tag); os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.join(HERE, "run.py"), "--seed", str(seed), "--train_seed", str(tseed),
               "--work", f"/tmp/dporep_{tag}", "--out", out] + extra
        env = dict(os.environ)
        if ngpu >= 2: env["CUDA_VISIBLE_DEVICES"] = str(k)
        log = open(os.path.join(out, "log.txt"), "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        print(f"started {tag} pid={p.pid} gpu={env.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        if ngpu < 2:  # sequential when only one device (or CPU)
            p.wait(); print(f"{tag} exit={p.returncode}", flush=True)
        else:
            procs.append((tag, p)); time.sleep(60)  # stagger startup (HF downloads, compile caches)
    for tag, p in procs:
        p.wait(); print(f"{tag} exit={p.returncode}", flush=True)
    for tag, _, _ in REPS:  # shrink each log (drop progress bars), print its tail into the job output
        try:
            fn = os.path.join(HERE, tag, "log.txt")
            keep = [l for l in open(fn, errors="replace") if "Processed prompts" not in l and "Capturing CUDA" not in l
                    and "it/s]" not in l][-400:]
            open(fn, "w").writelines(keep)
            print(f"--- {tag} log tail ---\n" + "".join(keep[-15:]), flush=True)
        except Exception as e: print(tag, "no log", e)
    os._exit(0)

if __name__ == "__main__":
    main()
