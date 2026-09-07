# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""vLLM hosting engine for the Model Playground.

Mirrors the proven ``LlamaCppBackend`` lifecycle: spawn ``vllm serve`` on an
ephemeral loopback port with a random per-instance API key, poll ``/health``
with early-crash detection, drain stdout to a log file, terminate cleanly on
stop. Unlike llama.cpp (one model at a time), this backend manages MULTIPLE
instances concurrently under a summed ``--gpu-memory-utilization`` budget (see
``utils/playground_settings.get_playground_gpu_budget``).

The process table lives in memory (this process spawned the servers); the DB
row is the durable status record and is reconciled against live PIDs at
startup (``sweep_stale_instances``) and on every status read.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import os
import secrets
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loggers import get_logger

from core.inference.vllm_args import ValidatedVllmArgs
from utils.paths import studio_root

logger = get_logger(__name__)

# vLLM model load can take minutes for large models / cold page cache.
_HEALTH_TIMEOUT_SECONDS = 1800
_HEALTH_INTERVAL_SECONDS = 1.0
_TERMINATE_TIMEOUT_SECONDS = 20.0


@dataclass
class _InstanceHandle:
    instance_id: str
    popen: subprocess.Popen
    port: int
    api_key: str
    base_url: str
    log_path: Path
    stop_lock: threading.Lock = field(default_factory = threading.Lock)


class VllmNotInstalledError(RuntimeError):
    """Raised when no ``vllm`` engine is importable in this installation."""


class VllmBackend:
    """Manages the lifecycle of hosted ``vllm serve`` subprocesses."""

    def __init__(self) -> None:
        self._handles: dict[str, _InstanceHandle] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @staticmethod
    def _find_vllm_cli() -> Optional[str]:
        """Locate a ``vllm`` CLI entry point."""
        # Prefer the venv that runs this backend, then PATH.
        candidates = []
        try:
            candidates.append(str(Path(sys_prefix()) / "bin" / "vllm"))
        except Exception:  # noqa: BLE001 -- best-effort discovery
            pass
        which = shutil.which("vllm")
        if which:
            candidates.append(which)
        for candidate in candidates:
            if candidate and os.access(candidate, os.X_OK):
                return candidate
        return None

    @classmethod
    def availability(cls) -> dict:
        """Detect whether vLLM can be used on this machine."""
        try:
            spec = importlib.util.find_spec("vllm")
        except Exception:  # noqa: BLE001 -- broken installs must not 500
            spec = None
        cli = cls._find_vllm_cli()
        version: Optional[str] = None
        if spec is not None:
            try:
                version = importlib.metadata.version("vllm")
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
        installed = spec is not None or cli is not None
        reason = None
        if not installed:
            reason = (
                "vLLM is not installed in this environment. Install it with "
                "`pip install vllm` on a Linux machine with a CUDA/ROCm GPU."
            )
        elif spec is None:
            reason = (
                "A vLLM CLI was found but the Python package is not importable "
                "from this environment."
            )
        return {
            "installed": installed,
            "version": version,
            "cli_path": cli,
            "importable": spec is not None,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start_instance(
        self, instance_id: str, model_path: str, model_slug: str, args: ValidatedVllmArgs
    ) -> _InstanceHandle:
        """Spawn ``vllm serve`` and block until it is healthy.

        Raises ``VllmNotInstalledError`` when no engine is available and
        ``RuntimeError`` when the server crashes or never becomes healthy.
        """
        availability = self.availability()
        if not availability["installed"]:
            raise VllmNotInstalledError(availability["reason"] or "vLLM is not installed")

        port = _free_port()
        api_key = secrets.token_urlsafe(32)
        logs_dir = studio_root() / "logs"
        logs_dir.mkdir(parents = True, exist_ok = True)
        log_path = logs_dir / f"vllm-{instance_id}.log"

        argv = [
            "vllm",
            "serve",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            api_key,
            "--served-model-name",
            model_slug,
            "--disable-log-requests",
        ] + args.build_cli_args()

        logger.info("Playground vLLM start: %s", " ".join(argv[:6] + ["..."]))
        with open(log_path, "ab") as log_file:
            log_file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} spawn ===\n".encode())
            log_file.flush()
            popen = subprocess.Popen(
                argv,
                stdout = log_file,
                stderr = subprocess.STDOUT,
                start_new_session = True,
            )
        # The child holds a dup of the fd; the parent's handle is closed by the
        # with-block and the log file stays appendable by name.

        handle = _InstanceHandle(
            instance_id = instance_id,
            popen = popen,
            port = port,
            api_key = api_key,
            base_url = f"http://127.0.0.1:{port}",
            log_path = log_path,
        )
        with self._lock:
            self._handles[instance_id] = handle

        try:
            self._wait_for_health(handle)
        except Exception:
            self._reap_process(instance_id)
            raise
        return handle

    def _wait_for_health(self, handle: _InstanceHandle) -> None:
        import httpx

        deadline = time.monotonic() + _HEALTH_TIMEOUT_SECONDS
        url = f"{handle.base_url}/health"
        while time.monotonic() < deadline:
            if handle.popen.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited during startup (code {handle.popen.returncode}). "
                    f"See {handle.log_path}"
                )
            try:
                response = httpx.get(url, timeout = 2.0, trust_env = False)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(_HEALTH_INTERVAL_SECONDS)
        raise RuntimeError(
            f"vLLM server did not become healthy within {_HEALTH_TIMEOUT_SECONDS}s. "
            f"See {handle.log_path}"
        )

    def stop_instance(self, instance_id: str) -> bool:
        """Terminate the server. Returns True if a live process was stopped."""
        with self._lock:
            handle = self._handles.pop(instance_id, None)
        if handle is None:
            return False
        with handle.stop_lock:
            if handle.popen.poll() is None:
                try:
                    # The whole process group (vLLM spawns engine-core children).
                    os.killpg(os.getpgid(handle.popen.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, AttributeError):
                    handle.popen.terminate()
                try:
                    handle.popen.wait(timeout = _TERMINATE_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(handle.popen.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, AttributeError):
                        handle.popen.kill()
                    handle.popen.wait(timeout = _TERMINATE_TIMEOUT_SECONDS)
            return True

    def stop_all(self) -> int:
        """Terminate every live instance (app shutdown)."""
        with self._lock:
            ids = list(self._handles.keys())
        stopped = 0
        for instance_id in ids:
            try:
                if self.stop_instance(instance_id):
                    stopped += 1
            except Exception:  # noqa: BLE001 -- shutdown must be best-effort
                logger.warning("Failed to stop playground instance %s", instance_id)
        return stopped

    def _reap_process(self, instance_id: str) -> None:
        with self._lock:
            self._handles.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Status / proxy info
    # ------------------------------------------------------------------

    def instance_alive(self, instance_id: str) -> bool:
        with self._lock:
            handle = self._handles.get(instance_id)
        return handle is not None and handle.popen.poll() is None

    def proxy_info(self, instance_id: str) -> Optional[dict]:
        """Connection info for proxying OpenAI requests to an instance."""
        with self._lock:
            handle = self._handles.get(instance_id)
        if handle is None or handle.popen.poll() is not None:
            return None
        return {
            "base_url": handle.base_url,
            "api_key": handle.api_key,
            "port": handle.port,
            "pid": handle.popen.pid,
            "log_path": str(handle.log_path),
        }

    def read_logs(
        self,
        instance_id: str,
        tail_bytes: int = 32768,
    ) -> Optional[str]:
        with self._lock:
            handle = self._handles.get(instance_id)
        path = handle.log_path if handle is not None else _known_log_path(instance_id)
        if path is None or not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - tail_bytes))
                return fh.read().decode("utf-8", errors = "replace")
        except OSError:
            return None

    def sweep_stale_instances(self) -> int:
        """Mark DB rows whose process is gone as stopped. Call at startup.

        Any DB row in a live status that this process does not own (fresh start
        after a crash) is stale: vLLM servers do not survive the backend.
        """
        from storage import playground_db

        stale = 0
        for record in playground_db.list_running_instances():
            instance_id = record["id"]
            if not self.instance_alive(instance_id):
                playground_db.update_instance(
                    instance_id,
                    status = "stopped",
                    stopped_at = _utcnow_iso(),
                    error = "Server not running (backend restarted or crashed).",
                )
                stale += 1
        return stale


def _known_log_path(instance_id: str) -> Optional[Path]:
    path = studio_root() / "logs" / f"vllm-{instance_id}.log"
    return path if path.exists() else None


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def sys_prefix() -> str:
    import sys
    return sys.prefix


_backend: Optional[VllmBackend] = None
_backend_lock = threading.Lock()


def get_vllm_backend() -> VllmBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = VllmBackend()
        return _backend


async def stop_all_instances_async() -> int:
    """Event-loop-friendly shutdown for all hosted instances."""
    backend = get_vllm_backend()
    return await asyncio.to_thread(backend.stop_all)
