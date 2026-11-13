"""Third format on MATH: for each math-g3 question (1.5B, positions 2000-2249), add one declarative-sympy sample (SYM)
and one sampled CoT (CS), both T=0.7, fresh seed. Resumable; appends rows to rows.jsonl."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m3-math"))
import run as M
SYS_SYM = ("You are a careful math solver. Do not compute the answer step by step yourself. Instead, translate the "
           "problem into SymPy: declare the unknowns with sympy.symbols, write every condition of the problem as a "
           "sympy.Eq (or an expression), and let sympy solve/simplify/evaluate them (sympy.solve, sympy.simplify, "
           "sympy.nsimplify, summation, etc.). Store the final numeric answer in a variable named `answer` and end "
           "with `print(answer)`. Output only the code in one ```python block.")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../math-g3/rows.jsonl"); ap.add_argument("--n", type=int, default=250)
ap.add_argument("--seed", type=int, default=7373); ap.add_argument("--out", default="rows.jsonl")
ap.add_argument("--max_new", type=int, default=1024)
a = ap.parse_args()
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
llm = Llama(hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=2048, n_threads=4, verbose=False, seed=a.seed)
df = M.load_math("test")
src = [json.loads(l) for l in open(a.src) if l.strip()][:a.n]
done = {json.loads(l)["idx"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()
def chat(s, q, seed):
    r = llm.create_chat_completion([{"role": "system", "content": s}, {"role": "user", "content": q}],
                                   temperature=0.7, max_tokens=a.max_new, seed=seed)
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"]
for j, r in enumerate(src):
    i = r["idx"]
    if i in done: continue
    t0 = time.time(); q = df.question[i]; g = float(df.gold[i]); assert abs(g - r["gold"]) < 1e-9
    st, sn = chat(SYS_SYM, q, a.seed * 10 + j)
    if "print(" not in st: st = st.replace("```\n", "\nprint(answer)\n```\n", 1) if st.count("```") >= 2 else st + "\nprint(answer)"
    sym = M.run_code(st)
    ct, cn = chat(M.SYS_COT, q, a.seed * 10 + 5000 + j); cs = M.extract(ct)
    row = dict(idx=i, gold=g, sym=sym, sym_ok=M.same(sym, g), sym_tok=sn, sym_code=st[-600:],
               cs=cs, cs_ok=M.same(cs, g), cs_boxed=M.boxed(ct) is not None, cs_tok=cn, cs_tail=ct[-300:],
               secs=round(time.time() - t0, 1))
    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    print(j, row["sym_ok"], row["cs_ok"], row["secs"], flush=True)
