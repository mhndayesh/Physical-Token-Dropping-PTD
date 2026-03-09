"""
OOM Boundary Test — runs each seq_len in a SEPARATE subprocess
so each test gets a fully clean GPU.
"""
import subprocess, sys, json
import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")

def test_single(seq_len, sparsity, layers=12):
    """Run a single forward+backward in a subprocess, return peak VRAM or 'OOM'."""
    code = f"""
import torch, gc
import sys
sys.path.insert(0, r'{SRC_DIR}')
from transformer_0_5b import SparseTransformer05B, Config
cfg = Config()
cfg.sparsity = {sparsity}
cfg.max_seq_len = {seq_len + 64}
cfg.n_layers = {layers}
cfg.block_size = 4
device = "cuda"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model = SparseTransformer05B(cfg).to(device).half()
dummy = torch.randint(0, cfg.vocab_size, (1, {seq_len}), device=device)
model.train()
out = model(dummy)
out.sum().backward()
peak = torch.cuda.max_memory_allocated() / (1024**2)
print(f"{{peak:.0f}}")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[-1])
        else:
            if "OutOfMemoryError" in result.stderr or "CUDA out of memory" in result.stderr:
                return "OOM"
            return f"ERR: {result.stderr[-100:]}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for oom_boundary_test.py")
    seq_lengths = [2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768]
    
    print("="*60)
    print("OOM BOUNDARY TEST (each test in a clean subprocess)")
    print("="*60)
    
    # Dense
    print("\n--- Dense (sparsity=1.0) ---")
    dense_results = {}
    dense_max = 0
    for sl in seq_lengths:
        print(f"  SeqLen {sl:>6}: ", end="", flush=True)
        r = test_single(sl, 1.0)
        if isinstance(r, float):
            print(f"{r:.0f} MB ✓")
            dense_results[sl] = r
            dense_max = sl
        else:
            print(f"*** {r} *** 💀")
            dense_results[sl] = r
            break
    
    # Sparse 30%
    print("\n--- Sparse 30% (sparsity=0.3) ---")
    sparse_results = {}
    sparse_max = 0
    for sl in seq_lengths:
        print(f"  SeqLen {sl:>6}: ", end="", flush=True)
        r = test_single(sl, 0.3)
        if isinstance(r, float):
            print(f"{r:.0f} MB ✓")
            sparse_results[sl] = r
            sparse_max = sl
        else:
            print(f"*** {r} *** 💀")
            sparse_results[sl] = r
            break
    
    # Summary
    print(f"\n{'─'*60}")
    print("RESULT")
    print(f"{'─'*60}")
    print(f"  Dense  max sequence length: {dense_max:,}")
    print(f"  Sparse max sequence length: {sparse_max:,}")
    if sparse_max > dense_max:
        print(f"  → Sparse handles {sparse_max/dense_max:.1f}x longer sequences!")
    
    print(f"\n{'SeqLen':>8}  {'Dense (MB)':>12}  {'Sparse (MB)':>12}  {'Saved':>8}")
    print(f"{'─'*8}  {'─'*12}  {'─'*12}  {'─'*8}")
    for sl in seq_lengths:
        d = dense_results.get(sl)
        s = sparse_results.get(sl)
        d_str = f"{d:.0f}" if isinstance(d, float) else str(d) if d else "—"
        s_str = f"{s:.0f}" if isinstance(s, float) else str(s) if s else "—"
        if isinstance(d, float) and isinstance(s, float):
            saved = f"{(1-s/d)*100:.0f}%"
        else:
            saved = "—"
        print(f"{sl:>8}  {d_str:>12}  {s_str:>12}  {saved:>8}")
    
    # Save
    json.dump({"dense": dense_results, "sparse": sparse_results,
               "dense_max": dense_max, "sparse_max": sparse_max},
              open("oom_results.json","w"), indent=2, default=str)
    print("\nSaved to oom_results.json")
