from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTD Phase 3: curriculum sparsity.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", default="data/tinystories_packed_qwen.pt")
    p.add_argument("--router-ckpt", default=None, help="Phase 2 checkpoint path")
    p.add_argument("--resume-ckpt", default=None, help="Phase 3 checkpoint path")
    p.add_argument("--schedule", default="0.99,0.9,0.7,0.5,0.3")
    p.add_argument("--steps-per-stage", type=int, default=2000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--block-size", type=int, default=6)
    p.add_argument("--segment-size", type=int, default=16)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--coverage-window",
        type=int,
        default=0,
        help="segments per local window for coverage penalty (0 disables)",
    )
    p.add_argument(
        "--coverage-weight",
        type=float,
        default=0.0,
        help="weight for coverage penalty (0 disables)",
    )
    p.add_argument(
        "--mask-loss",
        dest="mask_loss",
        action="store_true",
        default=True,
        help="optimize KL on routed/selected tokens (recommended)",
    )
    p.add_argument(
        "--full-loss",
        dest="mask_loss",
        action="store_false",
        help="optimize KL on all tokens (includes dropped-token penalty)",
    )
    return p.parse_args()


def parse_schedule(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Schedule is empty.")
    return vals


def get_batch(data: torch.Tensor, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, data.shape[0], (batch,))
    x = data[idx].to(device)
    inp = x[:, :-1]
    attn = torch.ones_like(inp, dtype=torch.bool)
    return inp, attn


def kl_distill(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    temperature: float,
    selection_mask: torch.Tensor | None = None,
    mask_loss: bool = False,
) -> torch.Tensor:
    s_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)

    eff_mask = token_mask
    if mask_loss and selection_mask is not None:
        eff_mask = eff_mask & selection_mask
    denom = eff_mask.float().sum().clamp_min(1.0)
    loss = (kl * eff_mask.float()).sum() / denom
    return loss * (temperature ** 2)


def coverage_penalty(segment_selection: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 0:
        return torch.zeros((), device=segment_selection.device, dtype=torch.float32)
    if segment_selection.numel() == 0:
        return torch.zeros((), device=segment_selection.device, dtype=torch.float32)
    n_blocks, bsz, n_seg = segment_selection.shape
    pad = (window - (n_seg % window)) % window
    if pad > 0:
        pad_tensor = torch.zeros(n_blocks, bsz, pad, dtype=torch.bool, device=segment_selection.device)
        segment_selection = torch.cat([segment_selection, pad_tensor], dim=-1)
        n_seg = segment_selection.size(-1)
    seg = segment_selection.view(n_blocks, bsz, n_seg // window, window)
    window_any = seg.any(dim=-1)
    missing = (~window_any).float().mean()
    return missing


def main() -> None:
    args = parse_args()
    schedule = parse_schedule(args.schedule)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Device: {device}, dtype={dtype}")
    teacher = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ptd_cfg = PTDConfig(
        block_size=args.block_size,
        segment_size=args.segment_size,
        keep_rate=schedule[0],
        drop_tokens=True,
        ste_gating=True,
    )
    student = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=ptd_cfg, torch_dtype=dtype).to(
        device=device, dtype=dtype
    )
    student.unfreeze_all()
    student.train()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Missing data file: {args.data}")
    data = torch.load(args.data, weights_only=True)
    print(f"Loaded data: {tuple(data.shape)}")

    optimizer = AdamW(student.parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)

    start_stage = 0
    stage_step = 0
    global_step = 0

    if args.router_ckpt:
        ckpt = torch.load(args.router_ckpt, map_location="cpu", weights_only=True)
        if "router_state" in ckpt:
            student.routers.load_state_dict(ckpt["router_state"], strict=True)
            print(f"Loaded router state: {args.router_ckpt}")
        elif "model_state" in ckpt:
            student.load_state_dict(ckpt["model_state"], strict=False)
            print(f"Loaded model state (fallback): {args.router_ckpt}")

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        student.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
        start_stage = int(ckpt["stage"])
        stage_step = int(ckpt["stage_step"])
        global_step = int(ckpt["global_step"])
        print(
            f"Resumed from {args.resume_ckpt} | stage={start_stage + 1} "
            f"stage_step={stage_step} global_step={global_step} lr={args.lr:g}"
        )

    t0 = time.time()
    for stage_idx, keep_rate in enumerate(schedule):
        if stage_idx < start_stage:
            continue
        student.set_keep_rate(keep_rate)
        student.set_drop_tokens(True)
        print(f"\nStage {stage_idx + 1}/{len(schedule)} | keep={keep_rate:.0%}")

        losses: List[float] = []
        losses_full: List[float] = []
        losses_sel: List[float] = []
        losses_cov: List[float] = []
        entropies: List[float] = []
        local_start = stage_step if stage_idx == start_stage else 0
        for step in range(local_start + 1, args.steps_per_stage + 1):
            global_step += 1
            inp, attn = get_batch(data, args.batch, device)
            with torch.no_grad():
                t_logits = teacher(input_ids=inp, attention_mask=attn).logits

            s_out, aux = student.forward_with_aux(input_ids=inp, attention_mask=attn)
            loss_full = kl_distill(
                student_logits=s_out.logits,
                teacher_logits=t_logits,
                token_mask=aux["token_mask"],
                temperature=args.temperature,
                selection_mask=aux["selection_mask"],
                mask_loss=False,
            )
            loss_sel = kl_distill(
                student_logits=s_out.logits,
                teacher_logits=t_logits,
                token_mask=aux["token_mask"],
                temperature=args.temperature,
                selection_mask=aux["selection_mask"],
                mask_loss=True,
            )
            loss = loss_sel if args.mask_loss else loss_full
            loss_cov = coverage_penalty(aux["segment_selection"], args.coverage_window)
            if args.coverage_weight > 0:
                loss = loss + args.coverage_weight * loss_cov

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            losses_full.append(loss_full.item())
            losses_sel.append(loss_sel.item())
            losses_cov.append(loss_cov.item())
            if aux["router_entropy"].numel() > 0:
                entropies.append(aux["router_entropy"].mean().item())
            stage_step = step

            if step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / args.log_every
                avg_full = sum(losses_full[-args.log_every:]) / args.log_every
                avg_sel = sum(losses_sel[-args.log_every:]) / args.log_every
                avg_cov = sum(losses_cov[-args.log_every:]) / args.log_every
                avg_ent = sum(entropies[-args.log_every:]) / max(1, len(entropies[-args.log_every:]))
                elapsed = time.time() - t0
                print(
                    f"  step {step:>5d}/{args.steps_per_stage} | "
                    f"loss {avg:.4f} | full {avg_full:.4f} | sel {avg_sel:.4f} | "
                    f"cov {avg_cov:.4f} | ent {avg_ent:.4f} | "
                    f"elapsed {elapsed:.1f}s | keep {keep_rate:.0%}"
                )

            if global_step % args.save_every == 0:
                path = f"checkpoints/ptd_v2_phase3_step{global_step:06d}.pt"
                torch.save(
                    {
                        "global_step": global_step,
                        "stage": stage_idx,
                        "stage_step": stage_step,
                        "keep_rate": keep_rate,
                        "schedule": schedule,
                        "ptd_config": student.ptd_config_dict(),
                        "model_state": student.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loss": losses[-1],
                    },
                    path,
                )
                print(f"  saved: {path}")

        stage_step = 0
        stage_path = f"checkpoints/ptd_v2_phase3_stage{stage_idx + 1}_keep{int(keep_rate * 100)}.pt"
        torch.save(
            {
                "global_step": global_step,
                "stage": stage_idx,
                "stage_step": 0,
                "keep_rate": keep_rate,
                "schedule": schedule,
                "ptd_config": student.ptd_config_dict(),
                "model_state": student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": losses[-1] if losses else None,
            },
            stage_path,
        )
        print(f"Stage complete: {stage_path}")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
