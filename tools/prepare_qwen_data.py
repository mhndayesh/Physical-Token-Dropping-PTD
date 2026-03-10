import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading TinyStories dataset (streaming)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    max_len = args.seq_len
    samples_to_prep = args.samples
    
    all_input_ids = []
    
    print(f"Tokenizing and packing {samples_to_prep} chunks of length {max_len}...")
    current_chunk = []
    
    pbar = tqdm(total=samples_to_prep)
    for item in dataset:
        text = item["text"]
        
        tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
        current_chunk.extend(tokens)
        
        while len(current_chunk) >= max_len:
            all_input_ids.append(current_chunk[:max_len])
            current_chunk = current_chunk[max_len:]
            pbar.update(1)
            
        if len(all_input_ids) >= samples_to_prep:
            break
    pbar.close()
            
    tensor_data = torch.tensor(all_input_ids[:samples_to_prep], dtype=torch.long)
    os.makedirs("data", exist_ok=True)
    torch.save(tensor_data, "data/tinystories_packed_qwen.pt")
    print(f"Saved {tensor_data.shape} to data/tinystories_packed_qwen.pt")

if __name__ == "__main__":
    main()
