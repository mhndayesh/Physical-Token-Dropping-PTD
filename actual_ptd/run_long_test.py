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
    p = argparse.ArgumentParser(description="Long-context test pipeline (dense + PTD).")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--prompt-file", default=None, help="prebuilt prompt.txt (context + question)")
    p.add_argument("--ideal-answer-file", default=None, help="prebuilt ideal_answer.txt")
    p.add_argument("--data-root", default=r"C:\new-arch-model\stress test\chats\100K")
    p.add_argument("--chat-id", default="1")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--question-set", default="abstention")
    p.add_argument("--question-index", type=int, default=0)
    p.add_argument("--device-map", default=None, help="set to auto to enable HF offload for dense")
    p.add_argument("--max-gpu-gb", type=int, default=None)
    p.add_argument("--max-cpu-gb", type=int, default=None)
    p.add_argument("--offload-folder", default="offload")
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--dense-use-cpu", action="store_true")
    p.add_argument("--ptd-use-cpu", action="store_true")
    p.add_argument("--report-json", default="long_test_report.json")
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


def build_tokens_from_prompt(
    tokenizer,
    prompt_text: str,
    answer_text: str,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    a_text = f" {answer_text.strip()}\n"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    a_tokens = tokenizer.encode(a_text, add_special_tokens=False)
    if seq_len <= len(a_tokens) + 1:
        raise ValueError("seq_len too small for answer length.")
    if len(prompt_tokens) + len(a_tokens) > seq_len:
        keep = seq_len - len(a_tokens)
        prompt_tokens = prompt_tokens[-keep:]
    total = prompt_tokens + a_tokens
    input_ids = torch.tensor(total[:-1], dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(total[1:], dtype=torch.long).unsqueeze(0)
    return input_ids, labels, len(a_tokens)


def ppl_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
    return math.exp(loss.item())


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, bool]:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float()
    acc = correct.mean().item()
    exact = bool(correct.all().item())
    return acc, exact


def eval_dense(
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    answer_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float | int | str | bool]:
    if args.device_map and args.offload_folder and device.type == "cuda":
        os.makedirs(args.offload_folder, exist_ok=True)
    print("Loading dense model...")
    t_load = time.time()
    if args.device_map and device.type == "cuda":
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
        dense = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    dense.eval()
    print(f"Dense model loaded in {time.time() - t_load:.1f}s")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_run = time.perf_counter()
    logits = dense(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        logits_to_keep=answer_len,
    ).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_run
    labels_ans = labels[:, -answer_len:]
    ppl = ppl_from_logits(logits, labels_ans)
    acc, exact = accuracy_from_logits(logits, labels_ans)
    metrics: Dict[str, float | int | str | bool] = {
        "mode": "dense",
        "answer_ppl": float(ppl),
        "answer_acc": float(acc),
        "answer_exact_match": bool(exact),
        "forward_sec": float(elapsed),
        "tokens_per_sec": float(input_ids.size(1) / max(elapsed, 1e-6)),
        "seq_len": int(input_ids.size(1)),
        "answer_len": int(answer_len),
    }
    if device.type == "cuda":
        metrics["max_cuda_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        metrics["max_cuda_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
    print(f"Dense forward in {elapsed:.3f}s | PPL {ppl:.3f} | acc {acc:.3f} | exact {exact}")
    del dense
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def eval_ptd(
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    answer_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float | int | str | bool]:
    print("Loading PTD model...")
    t_load = time.time()
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

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t_run = time.perf_counter()
    out, aux = sparse.forward_with_aux(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        logits_to_keep=answer_len,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_run
    labels_ans = labels[:, -answer_len:]
    logits = out.logits
    ppl_full = ppl_from_logits(logits, labels_ans)
    acc_full, exact_full = accuracy_from_logits(logits, labels_ans)

    sel_mask = aux["selection_mask"][:, 1:]
    sel_mask = sel_mask[:, -answer_len:]
    if sel_mask.any():
        sel_logits = logits[sel_mask]
        sel_labels = labels_ans[sel_mask]
        sel_loss = F.cross_entropy(sel_logits, sel_labels, reduction="mean")
        ppl_sel = math.exp(sel_loss.item())
        pred_sel = sel_logits.argmax(dim=-1)
        acc_sel = (pred_sel == sel_labels).float().mean().item()
    else:
        ppl_sel = float("inf")
        acc_sel = 0.0

    metrics: Dict[str, float | int | str | bool] = {
        "mode": "ptd",
        "keep_rate": float(args.keep_rate),
        "answer_ppl_selected": float(ppl_sel),
        "answer_ppl_full": float(ppl_full),
        "answer_acc_selected": float(acc_sel),
        "answer_acc_full": float(acc_full),
        "answer_exact_match_full": bool(exact_full),
        "forward_sec": float(elapsed),
        "tokens_per_sec": float(input_ids.size(1) / max(elapsed, 1e-6)),
        "seq_len": int(input_ids.size(1)),
        "answer_len": int(answer_len),
    }
    if device.type == "cuda":
        metrics["max_cuda_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        metrics["max_cuda_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))
    print(
        f"PTD forward in {elapsed:.3f}s | PPL(sel) {ppl_sel:.3f} | "
        f"PPL(full) {ppl_full:.3f} | acc(full) {acc_full:.3f}"
    )
    del sparse
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    args = parse_args()
    has_cuda = torch.cuda.is_available()
    if args.use_cpu:
        dense_device = torch.device("cpu")
        ptd_device = torch.device("cpu")
    else:
        dense_device = torch.device("cpu" if args.dense_use_cpu or not has_cuda else "cuda")
        ptd_device = torch.device("cpu" if args.ptd_use_cpu or not has_cuda else "cuda")

    dense_dtype = torch.bfloat16 if dense_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    ptd_dtype = torch.bfloat16 if ptd_device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if args.prompt_file:
        if not args.ideal_answer_file:
            raise ValueError("--ideal-answer-file is required when using --prompt-file.")
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Missing prompt file: {args.prompt_file}")
        if not os.path.exists(args.ideal_answer_file):
            raise FileNotFoundError(f"Missing answer file: {args.ideal_answer_file}")
        print("Using prompt file (no dataset load).")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
        with open(args.ideal_answer_file, "r", encoding="utf-8") as f:
            answer = f.read().strip()
        input_ids, labels, answer_len = build_tokens_from_prompt(
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            answer_text=answer,
            seq_len=args.seq_len,
        )
    else:
        chat_dir = os.path.join(args.data_root, str(args.chat_id))
        chat_path = os.path.join(chat_dir, "chat.json")
        probing_path = os.path.join(chat_dir, "probing_questions", "probing_questions.json")
        if not os.path.exists(chat_path):
            raise FileNotFoundError(f"Missing chat.json: {chat_path}")
        if not os.path.exists(probing_path):
            raise FileNotFoundError(f"Missing probing_questions.json: {probing_path}")
        chat_text = load_chat_text(chat_path)
        question, answer = load_question(probing_path, args.question_set, args.question_index)
        input_ids, labels, answer_len = build_tokens(
            tokenizer=tokenizer,
            chat_text=chat_text,
            question=question,
            answer=answer,
            seq_len=args.seq_len,
        )
    input_ids_dense = input_ids.to(dense_device)
    labels_dense = labels.to(dense_device)
    input_ids_ptd = input_ids.to(ptd_device)
    labels_ptd = labels.to(ptd_device)

    print(f"Total tokens (input): {input_ids.size(1)}")
    print(f"Answer tokens: {answer_len}")

    report: Dict[str, Dict[str, float | int | str | bool]] = {}
    print(f"Dense device: {dense_device}, dtype: {dense_dtype}")
    print(f"PTD device:   {ptd_device}, dtype: {ptd_dtype}")
    report["dense"] = eval_dense(args, input_ids_dense, labels_dense, answer_len, dense_device, dense_dtype)
    report["ptd"] = eval_ptd(args, input_ids_ptd, labels_ptd, answer_len, ptd_device, ptd_dtype)

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
