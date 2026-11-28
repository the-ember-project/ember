"""When will the queued Kaggle jobs fit? python self/tools/gpu_quota.py
Approximation: each use counts from its service_log timestamp for 7 days (matches the workflow's 'used in last 7 days')."""
import re, datetime as dt
LIM, now = 1440, dt.datetime.utcnow()
uses = [(dt.datetime.strptime(m[0], "%Y-%m-%d %H:%M"), int(m[1])) for m in
        re.findall(r"(\d{4}-\d\d-\d\d \d\d:\d\d) UTC Kaggle: ran .*?used about (\d+) GPU minutes", open("self/service_log.md").read())]
live = sorted((t + dt.timedelta(days=7), m) for t, m in uses if t + dt.timedelta(days=7) > now)
used = sum(m for _, m in live)
print(f"now {now:%m-%d %H:%M} used {used}/{LIM}, free {LIM-used}")
jobs = [l.split() for l in open("self/gpu_queue.txt") if l.strip() and not l.startswith("#")]
free, i = LIM - used, 0
for path, lim in jobs:
    need = int(lim) + 10
    when = now
    while free < need and i < len(live):
        when, m = live[i]; free += m; i += 1
    print(f"{path} needs {need}: " + (f"fits from ~{when:%m-%d %H:%M} UTC (next session after)" if free >= need else "never (limit too high)"))
    free -= int(lim)  # assume it uses about its limit (conservative)
