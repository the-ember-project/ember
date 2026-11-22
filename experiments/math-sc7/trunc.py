# Exploratory (session #38, review 2026-11-22): how much of SC7's gain over greedy (1024-token cap) sits on items
# where the greedy answer was truncated? Uses analyze.py's arms.
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): exec(open("analyze.py").read())
tr = [B[c["idx"]]["cot_tok"] >= 1024 for c in C]
print(f"greedy truncated (>=1024 tok) on {sum(tr)}/{n} items")
for lab, m in (("greedy truncated", tr), ("greedy finished", [not t for t in tr])):
    f = sum(a and not b and k for a, b, k in zip(A["SC7"], A["COT"], m)); b = sum(b and not a and k for a, b, k in zip(A["SC7"], A["COT"], m))
    print(f"SC7-COT on {lab} items: fixed {f} / broken {b} -> {100*(f-b)/n:+.1f} pts of the total (p={sign(f,b):.3g})")
for lab, m in (("greedy finished", [not t for t in tr]),):
    f = sum(a and not b and k for a, b, k in zip(A["SC7"], A["SC4"], m)); b = sum(b and not a and k for a, b, k in zip(A["SC7"], A["SC4"], m))
    print(f"SC7-SC4 on {lab} items: fixed {f} / broken {b} (p={sign(f,b):.3g})")
