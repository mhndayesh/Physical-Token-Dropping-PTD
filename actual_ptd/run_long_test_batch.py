from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch long-context test pipeline (dense + PTD).")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--data-root", default=r"C:\new-arch-model\stress test\chats\100K")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--question-set", default="abstention")
    p.add_argument("--max-questions", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--use-cpu", action="store_true")
    p.add_argument("--dense-use-cpu", action="store_true")
    p.add_argument("--ptd-use-cpu", action="store_true")
    p.add_argument("--device-map", default=None, help="set to auto to enable HF offload for dense")
    p.add_argument("--max-gpu-gb", type=int, default=None)
    p.add_argument("--max-cpu-gb", type=int, default=None)
    p.add_argument("--offload-folder", default="offload")
    p.add_argument("--report-json", default="long_test_batch.json")
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


def load_questions(probing_path: str, qset: str) -> List[Tuple[str, str]]:
    with open(probing_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[Dict[str, str]]] = json.load(f)
    if qset not in data or not data[qset]:
        raise ValueError(f"Question set '{qset}' not found in {probing_path}.")
    items = data[qset]
    out: List[Tuple[str, str]] = []
    for item in items:
        q = item.get("question", "").strip()
        a = item.get("ideal_response") or item.get("ideal_answer") or ""
        a = a.strip()
        if q and a:
            out.append((q, a))
    if not out:
        raise ValueError(f"No valid questions found in {probing_path}.")
    return out


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
        ctx_len = len(ctx_tokens)
    ctx_tokens = ctx_tokens[:ctx_len]
    total = ctx_tokens + q_tokens + a_tokens
    input_ids = torch.tensor(total[:-1], dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(total[1:], dtype=torch.long).unsqueeze(0)
    return input_ids, labels, len(a_tokens)


def ppl_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
    return math.exp(loss.item())


def acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, bool]:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).float()
    return correct.mean().item(), bool(correct.all().item())


def load_dense(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen2ForCausalLM:
    if args.device_map and args.offload_folder and device.type == "cuda":
        os.makedirs(args.offload_folder, exist_ok=True)
    if args.device_map and device.type == "cuda":
        max_memory = {}
        if args.max_gpu_gb:
            max_memory[0] = f"{args.max_gpu_gb}GiB"
        if args.max_cpu_gb:
            max_memory["cpu"] = f"{args.max_cpu_gb}GiB"
        model = Qwen2ForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=args.device_map,
            max_memory=max_memory or None,
            offload_folder=args.offload_folder,
            low_cpu_mem_usage=True,
        )
    else:
        model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    model.eval()
    return model


def load_ptd(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> PTDQwen2ForCausalLM:
    cfg = PTDConfig(keep_rate=args.keep_rate, drop_tokens=True)
    model = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=cfg, torch_dtype=dtype).to(
        device=device, dtype=dtype
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=True)
        elif "router_state" in ckpt:
            model.routers.load_state_dict(ckpt["router_state"], strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
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

    chat_ids = [
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    ]
    chat_ids = [d for d in chat_ids if d.isdigit()]
    if not chat_ids:
        raise ValueError(f"No numeric chat folders found in {args.data_root}.")

    samples: List[Tuple[str, str, str]] = []
    for chat_id in chat_ids:
        chat_dir = os.path.join(args.data_root, chat_id)
        chat_path = os.path.join(chat_dir, "chat.json")
        probing_path = os.path.join(chat_dir, "probing_questions", "probing_questions.json")
        if not os.path.exists(chat_path) or not os.path.exists(probing_path):
            continue
        questions = load_questions(probing_path, args.question_set)
        if not questions:
            continue
        q, a = random.choice(questions)
        samples.append((chat_id, q, a))
        if len(samples) >= args.max_questions:
            break

    if not samples:
        raise ValueError("No samples found.")

    print(f"Samples: {len(samples)}")
    print(f"Dense device: {dense_device}, dtype: {dense_dtype}")
    print(f"PTD device:   {ptd_device}, dtype: {ptd_dtype}")

    print("Loading dense model...")
    t_load = time.time()
    dense = load_dense(args, dense_device, dense_dtype)
    print(f"Dense loaded in {time.time() - t_load:.1f}s")

    print("Loading PTD model...")
    t_load = time.time()
    ptd = load_ptd(args, ptd_device, ptd_dtype)
    print(f"PTD loaded in {time.time() - t_load:.1f}s")

    results: List[Dict[str, float | int | str | bool]] = []

    for i, (chat_id, q, a) in enumerate(samples, 1):
        chat_dir = os.path.join(args.data_root, chat_id)
        chat_text = load_chat_text(os.path.join(chat_dir, "chat.json"))
        input_ids, labels, ans_len = build_tokens(tokenizer, chat_text, q, a, args.seq_len)

        input_ids_dense = input_ids.to(dense_device)
        labels_dense = labels.to(dense_device)
        input_ids_ptd = input_ids.to(ptd_device)
        labels_ptd = labels.to(ptd_device)

        with torch.no_grad():
            dense_logits = dense(
                input_ids=input_ids_dense,
                attention_mask=torch.ones_like(input_ids_dense, dtype=torch.bool),
                logits_to_keep=ans_len,
            ).logits
            dense_labels = labels_dense[:, -ans_len:]
            dense_ppl = ppl_from_logits(dense_logits, dense_labels)
            dense_acc, dense_exact = acc_from_logits(dense_logits, dense_labels)

            ptd_out, aux = ptd.forward_with_aux(
                input_ids=input_ids_ptd,
                attention_mask=torch.ones_like(input_ids_ptd, dtype=torch.bool),
                logits_to_keep=ans_len,
            )
            ptd_labels = labels_ptd[:, -ans_len:]
            ptd_logits = ptd_out.logits
            ptd_ppl_full = ppl_from_logits(ptd_logits, ptd_labels)
            ptd_acc_full, ptd_exact_full = acc_from_logits(ptd_logits, ptd_labels)

            sel_mask = aux["selection_mask"][:, 1:]
            sel_mask = sel_mask[:, -ans_len:]
            if sel_mask.any():
                sel_logits = ptd_logits[sel_mask]
                sel_labels = ptd_labels[sel_mask]
                sel_loss = F.cross_entropy(sel_logits, sel_labels, reduction="mean")
                ptd_ppl_sel = math.exp(sel_loss.item())
                ptd_acc_sel = (sel_logits.argmax(dim=-1) == sel_labels).float().mean().item()
            else:
                ptd_ppl_sel = float("inf")
                ptd_acc_sel = 0.0

        results.append(
            {
                "chat_id": chat_id,
                "answer_len": int(ans_len),
                "dense_ppl": float(dense_ppl),
                "dense_acc": float(dense_acc),
                "dense_exact": bool(dense_exact),
                "ptd_ppl_full": float(ptd_ppl_full),
                "ptd_acc_full": float(ptd_acc_full),
                "ptd_exact_full": bool(ptd_exact_full),
                "ptd_ppl_sel": float(ptd_ppl_sel),
                "ptd_acc_sel": float(ptd_acc_sel),
            }
        )
        print(
            f"[{i}/{len(samples)}] chat {chat_id} | dense ppl {dense_ppl:.2f} "
            f"| ptd ppl {ptd_ppl_sel:.2f} (sel)"
        )

    def avg(key: str) -> float:
        vals = [r[key] for r in results if isinstance(r.get(key), (float, int))]
        return float(sum(vals) / max(len(vals), 1))

    summary = {
        "n": len(results),
        "dense_ppl_mean": avg("dense_ppl"),
        "dense_acc_mean": avg("dense_acc"),
        "ptd_ppl_full_mean": avg("ptd_ppl_full"),
        "ptd_acc_full_mean": avg("ptd_acc_full"),
        "ptd_ppl_sel_mean": avg("ptd_ppl_sel"),
        "ptd_acc_sel_mean": avg("ptd_acc_sel"),
    }

    report = {"summary": summary, "results": results}
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
