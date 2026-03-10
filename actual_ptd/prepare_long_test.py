from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a long-context test pack.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data-root", default=r"C:\new-arch-model\stress test\chats\100K")
    p.add_argument("--chat-id", default="1")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--question-set", default="abstention")
    p.add_argument("--question-index", type=int, default=0)
    p.add_argument("--out-dir", default="long_context_test")
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


def build_prompt(
    tokenizer,
    chat_text: str,
    question: str,
    answer: str,
    seq_len: int,
) -> Tuple[str, str, Dict[str, int]]:
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

    prompt_text = tokenizer.decode(ctx_tokens) + q_text
    meta = {
        "seq_len": seq_len,
        "context_tokens": len(ctx_tokens),
        "question_tokens": len(q_tokens),
        "answer_tokens": len(a_tokens),
        "total_tokens": len(ctx_tokens) + len(q_tokens) + len(a_tokens),
    }
    return prompt_text, a_text.strip(), meta


def main() -> None:
    args = parse_args()
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
    prompt_text, ideal_answer, meta = build_prompt(
        tokenizer=tokenizer,
        chat_text=chat_text,
        question=question,
        answer=answer,
        seq_len=args.seq_len,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)
    with open(os.path.join(args.out_dir, "question.txt"), "w", encoding="utf-8") as f:
        f.write(question + "\n")
    with open(os.path.join(args.out_dir, "ideal_answer.txt"), "w", encoding="utf-8") as f:
        f.write(ideal_answer + "\n")
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote test pack to {args.out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
