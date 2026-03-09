"""
Full benchmark: All sparsity levels vs TRUE nn.TransformerEncoder baseline.
Each config in a separate subprocess for clean GPU state.
"""
import subprocess, sys, json

def bench_single(config_json):
    code = f"""
import torch, torch.nn as nn, time, json
cfg = json.loads('{config_json}')

if cfg['type'] == 'true_dense':
    class TrueDense(nn.Module):
        def __init__(self, d=1024, h=16, L=24, V=50257):
            super().__init__()
            self.emb = nn.Embedding(V, d)
            layer = nn.TransformerEncoderLayer(d, h, 4*d, batch_first=True, norm_first=True, activation='gelu')
            self.enc = nn.TransformerEncoder(layer, L)
            self.head = nn.Linear(d, V, bias=False)
            self.emb.weight = self.head.weight
        def forward(self, x):
            mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            return self.head(self.enc(self.emb(x), mask=mask, is_causal=True))
    model = TrueDense().cuda().half()
else:
    from transformer_0_5b import SparseTransformer05B, Config
    c = Config(); c.n_layers=24; c.sparsity=cfg['sparsity']
    model = SparseTransformer05B(c).cuda().half()

dummy = torch.randint(0, 50257, (1, 2048), device='cuda')
# VRAM
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
model.train(); out = model(dummy); out.sum().backward()
vram = torch.cuda.max_memory_allocated() / (1024**2)
# Latency
model.eval()
with torch.no_grad():
    for _ in range(3): model(dummy)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): model(dummy)
    torch.cuda.synchronize()
lat = (time.time()-t0)/5*1000
params = sum(p.numel() for p in model.parameters())/1e6
print(f'{{vram:.0f}}|{{lat:.1f}}|{{params:.1f}}')
"""
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, cwd=r"c:\udgrade_project_y")
        if r.returncode == 0:
            parts = r.stdout.strip().split('\n')[-1].split('|')
            return float(parts[0]), float(parts[1]), float(parts[2])
        return None, None, None
    except:
        return None, None, None

if __name__ == "__main__":
    configs = [
        {"name": "nn.TransformerEncoder (True Dense)", "type": "true_dense"},
        {"name": "Sparse 50%", "type": "sparse", "sparsity": 0.5},
        {"name": "Sparse 40%", "type": "sparse", "sparsity": 0.4},
        {"name": "Sparse 30%", "type": "sparse", "sparsity": 0.3},
        {"name": "Sparse 20%", "type": "sparse", "sparsity": 0.2},
        {"name": "Sparse 10%", "type": "sparse", "sparsity": 0.1},
    ]
    
    print("="*65)
    print("FULL BENCHMARK vs TRUE nn.TransformerEncoder (24L, d=1024, seq=2048)")
    print("Hardware: RTX 5070 12GB, i7-14700, 64GB DDR4")
    print("="*65)
    
    results = []
    for cfg in configs:
        print(f"  {cfg['name']:<40} ", end="", flush=True)
        vram, lat, params = bench_single(json.dumps(cfg))
        if vram:
            print(f"{vram/1024:.1f} GB | {lat:.1f} ms | {params:.0f}M params")
            results.append({"name": cfg["name"], "vram": vram, "lat": lat, "params": params})
        else:
            print("FAILED")
    
    # Summary
    dense = results[0]
    print(f"\n{'─'*65}")
    print(f"{'Model':<40} {'Latency':>8} {'Speedup':>8} {'VRAM':>8} {'Saved':>7}")
    print(f"{'─'*65}")
    for r in results:
        speedup = dense['lat'] / r['lat']
        saved = (1 - r['vram']/dense['vram']) * 100
        print(f"{r['name']:<40} {r['lat']:>6.1f}ms {speedup:>6.2f}x {r['vram']/1024:>6.1f}GB {saved:>+5.0f}%")
