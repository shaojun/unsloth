# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation for user-supplied ``vllm serve`` arguments.

Mirrors ``core/inference/llama_server_args.py``: users may pass extra runtime
flags, but the flags Unsloth owns (model, host, port, api key, served name)
are denied so a host request cannot hijack another port or impersonate another
instance. Args are passed to ``subprocess.Popen`` as a list, so there is no
shell-injection surface; this module only guards *semantic* safety.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Flags Unsloth sets itself; user values would break instance management.
DENIED_FLAGS = {
    "--model",
    "--host",
    "--port",
    "--api-key",
    "--served-model-name",
    "--help",
    "--version",
}

# Commonly useful flags surfaced in the UI form; free-form extras are still
# allowed (validated generically) for forward compatibility with new vLLM args.
KNOWN_VALUE_FLAGS = {
    "--max-model-len": int,
    "--gpu-memory-utilization": float,
    "--dtype": str,
    "--quantization": str,
    "--max-num-seqs": int,
    "--tensor-parallel-size": int,
    "--kv-cache-dtype": str,
    "--seed": int,
    "--swap-space": int,
    "--max-num-batched-tokens": int,
    "--enable-prefix-caching": bool,
    "--enforce-eager": bool,
    "--disable-log-requests": bool,
}

_ARG_NAME_RE = re.compile(r"^--[a-zA-Z0-9][a-zA-Z0-9_-]*$")

MAX_EXTRA_ARGS = 64
MAX_ARG_LENGTH = 256


class VllmArgsError(ValueError):
    """Raised when user-supplied vLLM args are unsafe or malformed."""


@dataclass
class ValidatedVllmArgs:
    """Structured runtime parameters + validated free-form extra args."""

    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    quantization: str | None = None
    max_num_seqs: int | None = None
    tensor_parallel_size: int | None = None
    kv_cache_dtype: str | None = None
    seed: int | None = None
    swap_space: int | None = None
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    extra_args: list[str] | None = None

    def to_json(self) -> dict:
        out: dict = {}
        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "dtype",
            "quantization",
            "max_num_seqs",
            "tensor_parallel_size",
            "kv_cache_dtype",
            "seed",
            "swap_space",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.enable_prefix_caching:
            out["enable_prefix_caching"] = True
        if self.enforce_eager:
            out["enforce_eager"] = True
        if self.extra_args:
            out["extra_args"] = list(self.extra_args)
        return out

    def build_cli_args(self) -> list[str]:
        """Render the validated parameters as ``vllm serve`` CLI arguments."""
        args: list[str] = []
        if self.max_model_len is not None:
            args += ["--max-model-len", str(self.max_model_len)]
        if self.gpu_memory_utilization is not None:
            args += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        if self.dtype:
            args += ["--dtype", self.dtype]
        if self.quantization:
            args += ["--quantization", self.quantization]
        if self.max_num_seqs is not None:
            args += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.tensor_parallel_size is not None and self.tensor_parallel_size > 1:
            args += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.kv_cache_dtype:
            args += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.seed is not None:
            args += ["--seed", str(self.seed)]
        if self.swap_space is not None:
            args += ["--swap-space", str(self.swap_space)]
        if self.enable_prefix_caching:
            args += ["--enable-prefix-caching"]
        if self.enforce_eager:
            args += ["--enforce-eager"]
        if self.extra_args:
            args += list(self.extra_args)
        return args


def _coerce_value(flag: str, raw: str, expected: type) -> object:
    try:
        if expected is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if expected is int:
            return int(raw)
        if expected is float:
            return float(raw)
        return raw
    except (TypeError, ValueError) as exc:
        raise VllmArgsError(f"{flag} expects a valid {expected.__name__}") from exc


def _validate_flag_value(flag: str, value: object) -> None:
    if flag == "--gpu-memory-utilization":
        if not 0.05 <= float(value) <= 1.0:  # type: ignore[arg-type]
            raise VllmArgsError("--gpu-memory-utilization must be between 0.05 and 1.0")
    if flag == "--tensor-parallel-size":
        if not 1 <= int(value) <= 8:  # type: ignore[arg-type]
            raise VllmArgsError("--tensor-parallel-size must be between 1 and 8")


def validate_extra_args(extra: list[str] | str | None) -> list[str]:
    """Validate a free-form list of extra CLI args (or a shell-ish string).

    Rules: every token must be a ``--flag`` or a plain value for the preceding
    flag; ``--flag=value`` form is split; denied flags raise; total count and
    length are bounded.
    """
    if not extra:
        return []
    tokens: list[str]
    if isinstance(extra, str):
        tokens = [t for t in extra.split() if t]
    else:
        tokens = []
        for item in extra:
            tokens.extend(t for t in str(item).split() if t)
    # Split --flag=value into two tokens for uniform handling.
    normalized: list[str] = []
    for token in tokens:
        if token.startswith("--") and "=" in token:
            name, _, value = token.partition("=")
            normalized += [name, value]
        else:
            normalized.append(token)
    if len(normalized) > MAX_EXTRA_ARGS * 2:
        raise VllmArgsError(f"Too many extra arguments (max {MAX_EXTRA_ARGS})")

    out: list[str] = []
    expecting_value = False
    current_flag = ""
    for token in normalized:
        if len(token) > MAX_ARG_LENGTH:
            raise VllmArgsError("Argument too long")
        if token.startswith("--"):
            if not _ARG_NAME_RE.match(token):
                raise VllmArgsError(f"Invalid argument name: {token}")
            if token in DENIED_FLAGS:
                raise VllmArgsError(f"{token} is managed by Unsloth and cannot be overridden")
            if expecting_value:
                raise VllmArgsError(f"Missing value for {current_flag}")
            if token in KNOWN_VALUE_FLAGS and KNOWN_VALUE_FLAGS[token] is bool:
                # Boolean flags take no value.
                out.append(token)
                expecting_value = False
                continue
            out.append(token)
            current_flag = token
            expecting_value = True
        else:
            if not expecting_value:
                raise VllmArgsError(f"Unexpected argument value: {token}")
            out.append(token)
            expecting_value = False
    if expecting_value:
        raise VllmArgsError(f"Missing value for {current_flag}")
    return out


def validate_vllm_args(
    structured: dict | None = None, extra_args: list[str] | str | None = None
) -> ValidatedVllmArgs:
    """Validate structured params + free-form extras into one args object."""
    structured = structured or {}
    parsed = ValidatedVllmArgs()

    if "max_model_len" in structured and structured["max_model_len"] is not None:
        value = _coerce_value("max_model_len", structured["max_model_len"], int)
        if value <= 0:
            raise VllmArgsError("max_model_len must be positive")
        parsed.max_model_len = value
    if "gpu_memory_utilization" in structured and structured["gpu_memory_utilization"] is not None:
        value = _coerce_value("gpu_memory_utilization", structured["gpu_memory_utilization"], float)
        _validate_flag_value("--gpu-memory-utilization", value)
        parsed.gpu_memory_utilization = value
    if "dtype" in structured and structured["dtype"]:
        dtype = str(structured["dtype"])
        if dtype not in {"auto", "float16", "bfloat16", "float32", "half"}:
            raise VllmArgsError(f"Unsupported dtype: {dtype}")
        parsed.dtype = dtype
    if "quantization" in structured and structured["quantization"]:
        parsed.quantization = str(structured["quantization"])
    if "max_num_seqs" in structured and structured["max_num_seqs"] is not None:
        value = _coerce_value("max_num_seqs", structured["max_num_seqs"], int)
        if value <= 0:
            raise VllmArgsError("max_num_seqs must be positive")
        parsed.max_num_seqs = value
    if "tensor_parallel_size" in structured and structured["tensor_parallel_size"] is not None:
        value = _coerce_value("tensor_parallel_size", structured["tensor_parallel_size"], int)
        _validate_flag_value("--tensor-parallel-size", value)
        parsed.tensor_parallel_size = value
    if "kv_cache_dtype" in structured and structured["kv_cache_dtype"]:
        kv = str(structured["kv_cache_dtype"])
        if kv not in {"auto", "fp8", "fp8_e5m2", "fp8_e4m3"}:
            raise VllmArgsError(f"Unsupported kv_cache_dtype: {kv}")
        parsed.kv_cache_dtype = kv
    if "seed" in structured and structured["seed"] is not None:
        parsed.seed = _coerce_value("seed", structured["seed"], int)
    if "swap_space" in structured and structured["swap_space"] is not None:
        parsed.swap_space = _coerce_value("swap_space", structured["swap_space"], int)
    if structured.get("enable_prefix_caching"):
        parsed.enable_prefix_caching = True
    if structured.get("enforce_eager"):
        parsed.enforce_eager = True

    # Free-form extras may re-specify structured keys (last-wins is fine) but
    # never the denied set.
    parsed.extra_args = validate_extra_args(extra_args)
    return parsed
