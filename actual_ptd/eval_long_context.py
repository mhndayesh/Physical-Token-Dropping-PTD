from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long-context PTD vs dense evaluation.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--mode", choices=["dense", "ptd"], default="dense")
    p.add_argument("--device-map", default=None, help="set to auto to enable HF offload")
    p.add_argument("--max-gpu-gb", type=int, default=None, help="max GPU memory for offload")
    p.add_argument("--max-cpu-gb", type=int, default=None, help="max CPU memory for offload")
    p.add_argument("--offload-folder", default="offload", help="folder for offloaded weights")
    p.add_argument("--data-root", default=r"C:\new-arch-model\stress test\chats\100K")
    p.add_argument("--chat-id", default="1", help="subfolder name under data-root")
    p.add_argument("--seq-len", type=int, default=16384, help="total tokens including question+answer")
    p.add_argument("--question-set", default="abstention")
    p.add_argument("--question-index", type=int, default=0)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--measure-speed", action="store_true", default=True)
    p.add_argument("--no-measure-speed", dest="measure_speed", action="store_false")
    p.add_argument("--measure-memory", action="store_true", default=True)
    p.add_argument("--no-measure-memory", dest="measure_memory", action="store_false")
    p.add_argument("--report-json", default=None, help="optional path to write metrics json")
    return p.parse_args()


def load_chat_text(chat_path: str) -> str:
    with open(chat_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines: List[str] = []
    for batch in data:
        turns = batch.get("turns", [])
        for convo in turns:
            for msg in convo:
                role = msg.get("role", "user").strip()
                content = msg.get("content", "").strip()
                if not content:
                    continue
                lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines) + "\n"


def load_question(probing_path: str, qset: str, qindex: int) -> Tuple[str, str]:
    with open(probing_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[Dict[str, str]]] = json.load(f)
    if qset not in data or not data[qset]:
        raise ValueError(f"Question set '{qset}' not found in {probing_path}.")
    items = data[qset]
    if qindex < 0 or qindex >= len(items):
        raise ValueError(f"Question index {qindex} out of range for set '{qset}'.")
    item = items[qindex]
    question = item.get("question", "").strip()
    answer = item.get("ideal_response") or item.get("ideal_answer") or ""
    answer = answer.strip()
    if not question or not answer:
        raise ValueError("Question or ideal answer is empty.")
    return question, answer


def build_tokens(
    tokenizer,
    chat_text: str,
    question: str,
    answer: str,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    q_text = f"User: {question}\nAssistant:"
    a_text = f" {answer}\n"
    q_tokens = tokenizer.encode(q_text, add_special_tokens=False)
    a_tokens = tokenizer.encode(a_text, add_special_tokens=False)

    if seq_len <= len(q_tokens) + len(a_tokens) + 1:
        raise ValueError("seq_len too small for question+answer length.")

    ctx_tokens = tokenizer.encode(chat_text, add_special_tokens=False)
    ctx_len = seq_len - len(q_tokens) - len(a_tokens)
    if len(ctx_tokens) < ctx_len:
        print(f"Warning: chat tokens {len(ctx_tokens)} < context length {ctx_len}. Using full chat.")
        ctx_len = len(ctx_tokens)
    ctx_tokens = ctx_tokens[:ctx_len]

    total = ctx_tokens + q_tokens + a_tokens
    input_ids = torch.tensor(total[:-1], dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(total[1:], dtype=torch.long).unsqueeze(0)
    return input_ids, labels, len(a_tokens)


def ppl_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
    return math.exp(loss.item())


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    chat_dir = os.path.join(args.data_root, str(args.chat_id))
    chat_path = os.path.join(chat_dir, "chat.json")
    probing_path = os.path.join(chat_dir, "probing_questions", "probing_questions.json")
    if not os.path.exists(chat_path):
        raise FileNotFoundError(f"Missing chat.json: {chat_path}")
    if not os.path.exists(probing_path):
        raise FileNotFoundError(f"Missing probing_questions.json: {probing_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    chat_text = load_chat_text(chat_path)
    question, answer = load_question(probing_path, args.question_set, args.question_index)
    input_ids, labels, answer_len = build_tokens(
        tokenizer=tokenizer,
        chat_text=chat_text,
        question=question,
        answer=answer,
        seq_len=args.seq_len,
    )
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    print(f"Total tokens (input): {input_ids.size(1)}")
    print(f"Answer tokens: {answer_len}")

    if args.device_map and args.offload_folder:
        os.makedirs(args.offload_folder, exist_ok=True)

    metrics: Dict[str, float | str | int] = {}

    with torch.no_grad():
        if args.mode == "dense":
            t_load = time.time()
            print("Loading dense model...")
            if args.device_map:
                max_memory = {}
                if args.max_gpu_gb:
                    max_memory[0] = f"{args.max_gpu_gb}GiB"
                if args.max_cpu_gb:
                    max_memory["cpu"] = f"{args.max_cpu_gb}GiB"
                dense = Qwen2ForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=dtype,
                    device_map=args.device_map,
                    max_memory=max_memory or None,
                    offload_folder=args.offload_folder,
                    low_cpu_mem_usage=True,
                )
            else:
                dense = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(
                    device=device, dtype=dtype
                )
            dense.eval()
            print(f"Dense model loaded in {time.time() - t_load:.1f}s")
            if args.measure_memory and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if args.measure_speed and device.type == "cuda":
                torch.cuda.synchronize()
            t_run = time.perf_counter()
            dense_logits = dense(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                logits_to_keep=answer_len,
            ).logits
            if args.measure_speed and device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_run
            dense_labels = labels[:, -answer_len:]
            dense_ppl = ppl_from_logits(dense_logits, dense_labels)
            print(f"Dense forward in {elapsed:.3f}s")
            print(f"Dense answer PPL : {dense_ppl:.3f}")
            metrics["mode"] = "dense"
            metrics["answer_ppl"] = float(dense_ppl)
            metrics["seq_len"] = int(input_ids.size(1))
            metrics["answer_len"] = int(answer_len)
            if args.measure_speed:
                metrics["forward_sec"] = float(elapsed)
                metrics["tokens_per_sec"] = float(input_ids.size(1) / max(elapsed, 1e-6))
            if args.measure_memory and device.type == "cuda":
                metrics["max_cuda_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                metrics["max_cuda_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
        else:
            t_load = time.time()
            print("Loading PTD model...")
            ptd_cfg = PTDConfig(keep_rate=args.keep_rate, drop_tokens=True)
            sparse = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=ptd_cfg, torch_dtype=dtype).to(
                device=device, dtype=dtype
            )
            if args.checkpoint:
                ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
                if "model_state" in ckpt:
                    sparse.load_state_dict(ckpt["model_state"], strict=True)
                elif "router_state" in ckpt:
                    sparse.routers.load_state_dict(ckpt["router_state"], strict=True)
            sparse.eval()
            print(f"PTD model loaded in {time.time() - t_load:.1f}s")

            if args.measure_memory and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            if args.measure_speed and device.type == "cuda":
                torch.cuda.synchronize()
            t_run = time.perf_counter()
            ptd_out, aux = sparse.forward_with_aux(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                logits_to_keep=answer_len,
            )
            if args.measure_speed and device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_run
            ptd_labels = labels[:, -answer_len:]
            ptd_logits = ptd_out.logits
            full_ppl = ppl_from_logits(ptd_logits, ptd_labels)

            sel_mask = aux["selection_mask"][:, 1:]
            sel_mask = sel_mask[:, -answer_len:]
            if sel_mask.any():
                sel_logits = ptd_logits[sel_mask]
                sel_labels = ptd_labels[sel_mask]
                sel_loss = F.cross_entropy(sel_logits, sel_labels, reduction="mean")
                sel_ppl = math.exp(sel_loss.item())
            else:
                sel_ppl = float("inf")

            print(f"PTD forward in {elapsed:.3f}s")
            print(f"PTD answer PPL   : {sel_ppl:.3f} (selected-token)")
            print(f"PTD answer PPL   : {full_ppl:.3f} (full-token)")
            metrics["mode"] = "ptd"
            metrics["answer_ppl_selected"] = float(sel_ppl)
            metrics["answer_ppl_full"] = float(full_ppl)
            metrics["seq_len"] = int(input_ids.size(1))
            metrics["answer_len"] = int(answer_len)
            metrics["keep_rate"] = float(args.keep_rate)
            if args.measure_speed:
                metrics["forward_sec"] = float(elapsed)
                metrics["tokens_per_sec"] = float(input_ids.size(1) / max(elapsed, 1e-6))
            if args.measure_memory and device.type == "cuda":
                metrics["max_cuda_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                metrics["max_cuda_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
