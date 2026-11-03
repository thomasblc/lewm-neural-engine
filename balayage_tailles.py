import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import banc as B
for c in (64, 96, 128, 256):
    B.CACHE = c
    print(f"=== cache {c} : {B.n_poids()} poids, budget gardé {B.budget_gardes()}", flush=True)
    t = time.time()
    B.cmd_preparer(None)
    print(f"    ({time.time() - t:.0f} s)", flush=True)
print("FINI", flush=True)
