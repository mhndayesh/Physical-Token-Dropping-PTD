import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from transformer_0_5b import SparseTransformer05B, Config
import math, time, pandas as pd

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path, num_samples=500):
        self.data = torch.load(data_path, weights_only=False)[:num_samples]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]

class TrueDense(nn.Module):
    def __init__(self, d=256, h=4, L=4, V=50257):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        layer = nn.TransformerEncoderLayer(d, h, 4*d, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, L)
        self.head = nn.Linear(d, V, bias=False)
        self.emb.weight = self.head.weight
    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.head(self.enc(self.emb(x), mask=mask, is_causal=True))

def train_and_eval(model_type, sparsity=0.3, steps=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)  # For consistent initialization
    
    if model_type == "true_dense":
        model = TrueDense(d=256, h=4, L=4).to(device)
    else:
        config = Config()
        config.d_model = 256; config.n_heads = 4; config.n_layers = 4
        config.sparsity = sparsity; config.max_seq_len = 512
        model = SparseTransformer05B(config).to(device)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    dataset = TinyStoriesDataset("tinystories_tokenized.pt")
    
    # manual seed again for consistent dataloader shuffling across runs
    generator = torch.Generator().manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)

    model.train()
    step = 0
    t0 = time.time()
    for split in range(10): # dummy loop
        for x, y in dataloader:
            if step >= steps: break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, 50257), y.view(-1))
            loss.backward()
            optimizer.step()
            step += 1
        if step >= steps: break

    t_train = time.time() - t0
    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(20):
            x, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = model(x)
            total_loss += criterion(logits.view(-1, 50257), y.view(-1)).item()
    
    ppl = math.exp(total_loss / 20)
    return ppl, t_train

if __name__ == "__main__":
    configs = [
        {"name": "PyTorch True Dense", "type": "true_dense", "s": 1.0},
        {"name": "PTD (100% tokens)", "type": "sparse", "s": 1.0},
        {"name": "PTD (30% tokens)", "type": "sparse", "s": 0.3},
    ]
    
    print("="*60)
    print("ACCURACY TEST vs TRUE PYTORCH DENSE (200 steps)")
    print("="*60)
    
    results = []
    
    for cfg in configs:
        print(f"Training {cfg['name']}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ppl, t_train = train_and_eval(cfg['type'], cfg['s'])
        results.append({"Model": cfg["name"], "PPL": round(ppl, 2), "Train Time (s)": round(t_train, 1)})
        print(f"  -> PPL: {ppl:.2f} | Time: {t_train:.1f}s")
        
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)
    print(df.to_string(index=False))
