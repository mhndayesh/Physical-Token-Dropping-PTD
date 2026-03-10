from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoConfig


DEFAULT_PTD_CONFIG: Dict[str, Any] = {
    "block_size": 6,
    "segment_size": 16,
    "keep_rate": 0.7,
    "keep_rates": None,
    "router_rank": 16,
    "router_queries": 8,
    "router_type": "mq",
    "router_dim": 128,
    "router_heads": 2,
    "router_layers": 1,
    "router_jitter": 0.01,
    "drop_tokens": True,
    "ste_gating": True,
}


LOADER_SOURCE = """from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from model import PTDConfig, PTDQwen2ForCausalLM

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


def _resolve_repo_path(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.exists():
        return p
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is required to load from repo id. "
            "Install it or pass a local package path."
        )
    local = snapshot_download(
        repo_id=path_or_repo,
        allow_patterns=[
            "*.json",
            "*.md",
            "*.pt",
            "model.py",
            "__init__.py",
            "hf_ptd_loader.py",
            "requirements.txt",
            ".gitattributes",
        ],
    )
    return Path(local)


def _resolve_dtype(dtype: str) -> torch.dtype:
    dt = dtype.lower()
    if dt == "bfloat16":
        return torch.bfloat16
    if dt == "float16":
        return torch.float16
    if dt == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_ptd_model(
    path_or_repo: str,
    *,
    device: str | None = None,
    dtype: str = "bfloat16",
    keep_rate: float | None = None,
) -> Tuple[PTDQwen2ForCausalLM, Dict[str, Any]]:
    pkg_dir = _resolve_repo_path(path_or_repo)
    cfg_path = pkg_dir / "ptd_package_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing package config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        package_cfg = json.load(f)

    ptd_cfg = PTDConfig(**package_cfg["ptd_config"])
    torch_dtype = _resolve_dtype(dtype)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = PTDQwen2ForCausalLM.from_pretrained(
        package_cfg["base_model"],
        ptd_config=ptd_cfg,
        torch_dtype=torch_dtype,
    )

    package_type = package_cfg["package_type"]
    if package_type == "router_only":
        router_state = torch.load(pkg_dir / "router_state.pt", map_location="cpu", weights_only=True)
        model.routers.load_state_dict(router_state, strict=True)
    elif package_type == "full_state":
        model_state = torch.load(pkg_dir / "ptd_model_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(model_state, strict=True)
    else:
        raise ValueError(f"Unsupported package type: {package_type}")

    target_keep = keep_rate if keep_rate is not None else package_cfg.get("recommended_keep_rate")
    if target_keep is not None:
        model.set_keep_rate(float(target_keep))

    model = model.to(device=resolved_device, dtype=torch_dtype)
    model.eval()
    return model, package_cfg
"""


HF_CONFIG_SOURCE = """from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class PTDQwen2Config(PretrainedConfig):
    model_type = "ptd_qwen2"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        tokenizer: Optional[str] = None,
        ptd_config: Optional[Dict[str, Any]] = None,
        package_type: str = "full_state",
        recommended_keep_rate: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = base_model
        self.tokenizer = tokenizer or base_model
        self.ptd_config = ptd_config or {}
        self.package_type = package_type
        self.recommended_keep_rate = float(recommended_keep_rate)
"""


HF_MODEL_SOURCE = """from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from transformers import PreTrainedModel

from .configuration_ptd_qwen2 import PTDQwen2Config
from .model import PTDConfig, PTDQwen2ForCausalLM as _CorePTDModel

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


def _resolve_torch_dtype(torch_dtype: Any) -> torch.dtype:
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype is None:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    s = str(torch_dtype).lower()
    if "bfloat16" in s:
        return torch.bfloat16
    if "float16" in s:
        return torch.float16
    if "float32" in s:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


class PTDQwen2ForCausalLM(PreTrainedModel):
    config_class = PTDQwen2Config
    main_input_name = "input_ids"
    _supports_cache_class = False

    def __init__(self, config: PTDQwen2Config) -> None:
        super().__init__(config)
        self.ptd_model: Optional[_CorePTDModel] = None

    def _supports_default_dynamic_cache(self) -> bool:
        return False

    def _supports_static_cache(self) -> bool:
        return False

    def _supports_quantized_cache(self) -> bool:
        return False

    @staticmethod
    def _resolve_repo_dir(path_or_repo: str, local_files_only: bool = False) -> Path:
        p = Path(path_or_repo)
        if p.exists():
            return p
        if snapshot_download is None:
            raise RuntimeError(
                "huggingface_hub is required to load from repo id. "
                "Install huggingface_hub or pass a local path."
            )
        local = snapshot_download(
            repo_id=path_or_repo,
            local_files_only=local_files_only,
            allow_patterns=[
                "*.json",
                "*.md",
                "*.pt",
                "*.py",
                ".gitattributes",
            ],
        )
        return Path(local)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any):
        keep_rate = kwargs.pop("keep_rate", None)
        device = kwargs.pop("device", None)
        device_map = kwargs.pop("device_map", None)
        local_files_only = bool(kwargs.pop("local_files_only", False))
        torch_dtype = _resolve_torch_dtype(kwargs.pop("torch_dtype", None))

        repo_dir = cls._resolve_repo_dir(pretrained_model_name_or_path, local_files_only=local_files_only)
        config = PTDQwen2Config.from_pretrained(str(repo_dir), trust_remote_code=True)
        model = cls(config)

        ptd_cfg = PTDConfig(**config.ptd_config)
        core = _CorePTDModel.from_pretrained(
            config.base_model,
            ptd_config=ptd_cfg,
            torch_dtype=torch_dtype,
        )

        if config.package_type == "full_state":
            state = torch.load(repo_dir / "ptd_model_state.pt", map_location="cpu", weights_only=True)
            core.load_state_dict(state, strict=True)
        elif config.package_type == "router_only":
            r_state = torch.load(repo_dir / "router_state.pt", map_location="cpu", weights_only=True)
            core.routers.load_state_dict(r_state, strict=True)
        else:
            raise ValueError(f"Unsupported package_type: {config.package_type}")

        target_keep = config.recommended_keep_rate if keep_rate is None else float(keep_rate)
        core.set_keep_rate(target_keep)

        if device is None:
            if isinstance(device_map, str) and device_map == "cpu":
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

        core = core.to(device=device, dtype=torch_dtype)
        core.eval()
        model.ptd_model = core
        return model

    def _ensure_loaded(self) -> _CorePTDModel:
        if self.ptd_model is None:
            raise RuntimeError("Model is not initialized. Use from_pretrained().")
        return self.ptd_model

    @property
    def device(self) -> torch.device:
        return self._ensure_loaded().device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        core = self._ensure_loaded()
        kwargs.setdefault("ptd_use_sparse_cache", True)
        return core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        model_inputs.update(kwargs)
        return model_inputs
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PTD checkpoint as Hugging Face package folder.")
    p.add_argument("--checkpoint", required=True, help="PTD checkpoint path (.pt)")
    p.add_argument("--out-dir", required=True, help="Output folder for HF package")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B", help="Base model id/path for loader")
    p.add_argument("--tokenizer", default=None, help="Tokenizer id/path (defaults to base model)")
    p.add_argument(
        "--package-type",
        choices=["full_state", "router_only"],
        default="full_state",
        help="full_state = exact phase3 model; router_only = small adapter-style router package",
    )
    p.add_argument("--keep-rate", type=float, default=0.7, help="Recommended keep rate to store in package config")
    p.add_argument("--model-label", default="Qwen2.5-0.5B PTD keep70", help="Human-readable label in README")
    return p.parse_args()


def _merge_ptd_config(raw: Dict[str, Any] | None, keep_rate: float) -> Dict[str, Any]:
    cfg = dict(DEFAULT_PTD_CONFIG)
    if raw:
        cfg.update(raw)
    cfg["keep_rate"] = keep_rate
    return cfg


def _extract_router_state(model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    router_state: Dict[str, torch.Tensor] = {}
    for k, v in model_state.items():
        if k.startswith("routers."):
            router_state[k.replace("routers.", "", 1)] = v
    if not router_state:
        raise RuntimeError("No router parameters found in model_state.")
    return router_state


def _write_readme(out_dir: Path, package_cfg: Dict[str, Any], ckpt_name: str) -> None:
    package_type = package_cfg["package_type"]
    readme = textwrap.dedent(
        f"""\
        # {package_cfg["model_label"]}

        PTD package exported from `{ckpt_name}`.

        - Base model: `{package_cfg["base_model"]}`
        - Model scale: `0.5B`
        - Package type: `{package_type}`
        - Recommended keep rate: `{package_cfg["recommended_keep_rate"]}`
        - PTD config: see `ptd_package_config.json`

        ## Files

        - `ptd_package_config.json`: metadata and PTD config.
        - `ptd_model_state.pt` or `router_state.pt`: PTD weights (depends on package type).
        - `model.py`: PTD runtime model implementation.
        - `hf_ptd_loader.py`: loader for local path or HF repo id.
        - `requirements.txt`: minimal runtime deps.

        ## Quick Start

        ```python
        from hf_ptd_loader import load_ptd_model

        model, meta = load_ptd_model(
            ".",
            device="cuda",
            dtype="bfloat16",
            keep_rate={package_cfg["recommended_keep_rate"]},
        )
        print(meta["model_label"])
        ```

        ## Notes

        - `full_state` package reproduces your trained PTD checkpoint.
        - `router_only` package is much smaller, but does **not** include full phase3 backbone tuning.
        """
    ).strip() + "\n"
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def _write_requirements(out_dir: Path) -> None:
    req = "torch>=2.1\ntransformers>=4.45\nhuggingface_hub>=0.23\n"
    (out_dir / "requirements.txt").write_text(req, encoding="utf-8")


def _write_gitattributes(out_dir: Path) -> None:
    attrs = "*.pt filter=lfs diff=lfs merge=lfs -text\n"
    (out_dir / ".gitattributes").write_text(attrs, encoding="utf-8")


def _write_hf_config_json(out_dir: Path, package_cfg: Dict[str, Any]) -> None:
    base_cfg = AutoConfig.from_pretrained(package_cfg["base_model"]).to_dict()
    cfg = {
        **base_cfg,
        "model_type": "ptd_qwen2",
        "architectures": ["PTDQwen2ForCausalLM"],
        "base_model": package_cfg["base_model"],
        "tokenizer": package_cfg["tokenizer"],
        "package_type": package_cfg["package_type"],
        "recommended_keep_rate": package_cfg["recommended_keep_rate"],
        "ptd_config": package_cfg["ptd_config"],
        "auto_map": {
            "AutoConfig": "configuration_ptd_qwen2.PTDQwen2Config",
            "AutoModelForCausalLM": "modeling_ptd_qwen2.PTDQwen2ForCausalLM",
        },
    }
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_hf_remote_code(out_dir: Path) -> None:
    (out_dir / "configuration_ptd_qwen2.py").write_text(HF_CONFIG_SOURCE, encoding="utf-8")
    (out_dir / "modeling_ptd_qwen2.py").write_text(HF_MODEL_SOURCE, encoding="utf-8")

    init_code = (
        "from .configuration_ptd_qwen2 import PTDQwen2Config\n"
        "from .modeling_ptd_qwen2 import PTDQwen2ForCausalLM\n\n"
        "__all__ = [\"PTDQwen2Config\", \"PTDQwen2ForCausalLM\"]\n"
    )
    (out_dir / "__init__.py").write_text(init_code, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ptd_cfg = _merge_ptd_config(ckpt.get("ptd_config"), keep_rate=args.keep_rate)

    model_state = ckpt.get("model_state")
    router_state = ckpt.get("router_state")

    if args.package_type == "full_state":
        if model_state is None:
            raise RuntimeError("Checkpoint has no model_state. Use --package-type router_only instead.")
        torch.save(model_state, out_dir / "ptd_model_state.pt")
    else:
        if router_state is not None:
            torch.save(router_state, out_dir / "router_state.pt")
        elif model_state is not None:
            extracted = _extract_router_state(model_state)
            torch.save(extracted, out_dir / "router_state.pt")
        else:
            raise RuntimeError("Checkpoint has neither router_state nor model_state.")

    package_cfg = {
        "format_version": 1,
        "package_type": args.package_type,
        "base_model": args.base_model,
        "tokenizer": args.tokenizer or args.base_model,
        "model_label": args.model_label,
        "model_scale": "0.5B",
        "recommended_keep_rate": float(args.keep_rate),
        "ptd_config": ptd_cfg,
        "source_checkpoint": str(ckpt_path.as_posix()),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    (out_dir / "ptd_package_config.json").write_text(
        json.dumps(package_cfg, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    shutil.copy2(Path("actual_ptd") / "model.py", out_dir / "model.py")
    (out_dir / "hf_ptd_loader.py").write_text(LOADER_SOURCE, encoding="utf-8")
    _write_hf_remote_code(out_dir)
    _write_hf_config_json(out_dir, package_cfg)
    _write_requirements(out_dir)
    _write_gitattributes(out_dir)
    _write_readme(out_dir, package_cfg, ckpt_name=ckpt_path.name)

    print(f"Export complete: {out_dir}")
    print(f"Package type: {args.package_type}")
    if args.package_type == "full_state":
        print(f"Weights file: {out_dir / 'ptd_model_state.pt'}")
    else:
        print(f"Weights file: {out_dir / 'router_state.pt'}")


if __name__ == "__main__":
    main()
