"""Setup test: does the Kaggle job see a GPU, and do result files come back?"""
import json, time
import torch

ok = torch.cuda.is_available()
print("cuda available:", ok)
info = {"cuda": ok, "torch": torch.__version__}
if ok:
    p = torch.cuda.get_device_properties(0)
    info.update(gpu=p.name, memory_gb=round(p.total_memory / 1e9, 1))
    a = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        b = a @ a
    torch.cuda.synchronize()
    info["matmul_4096_seconds"] = round((time.time() - t) / 10, 4)
print(json.dumps(info, indent=2))
json.dump(info, open("result.json", "w"), indent=2)
