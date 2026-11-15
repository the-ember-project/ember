"""Session #24 (review 2026-11-15): how many PAL samples are the demo simulator with only the action list swapped?
Normalise: drop the `steps = [...]` line, comments, blank lines; compare the multiset of remaining lines to the demo's."""
import json, re, sys
sys.path.insert(0, '.'); from run import EX_PAL
def norm(t):
    m = re.search(r"```python\n(.*?)```", t, re.S); code = m.group(1) if m else t
    ls = [re.sub(r"#.*", "", l).rstrip() for l in code.splitlines()]
    return sorted(l for l in ls if l.strip() and not l.strip().startswith("steps ="))
D = norm(EX_PAL); n = same = same_ok = ok = 0
for f in ["pal_s.jsonl"]:
    for l in open(f):
        r = json.loads(l)
        for s in r["samples"]:
            n += 1; c = s["pred"] == r["gold"]; ok += c
            if norm(s["text"]) == D: same += 1; same_ok += c
print(f"PAL samples {n}: demo simulator verbatim (only steps swapped, branch order free) {same} ({100*same/n:.1f}%); "
      f"acc on those {100*same_ok/max(1,same):.1f}%, on the rest {100*(ok-same_ok)/max(1,n-same):.1f}%")
