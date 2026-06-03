from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_LLAMACLI_PATH = BASE_DIR / "llama.cpp" / "build" / "bin" / "llama-cli"

# Shell wrapper path — solves LD_LIBRARY_PATH for subprocess reliably
WRAPPER_SCRIPT = BASE_DIR / "run_llama.sh"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("llama_inference_server")


# ── Write wrapper script on first import ────────────────────────────────────
def _ensure_wrapper(llama_cli_path: Path) -> Path:
    """
    Write a shell wrapper that sets LD_LIBRARY_PATH before exec-ing llama-cli.
    This is the most reliable way to inject env vars for a subprocess — Python's
    env= dict works for most cases but the dynamic linker on Jetson reads
    LD_LIBRARY_PATH at exec time, so we need it set in the shell that calls exec.
    """
    script = f"""#!/bin/bash
export LD_LIBRARY_PATH=/home/nvidia/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
exec {llama_cli_path} "$@"
"""
    WRAPPER_SCRIPT.write_text(script)
    WRAPPER_SCRIPT.chmod(0o755)
    logger.info("wrapper script written: %s", WRAPPER_SCRIPT)
    return WRAPPER_SCRIPT


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    sensor: dict[str, Any] | None = None


@dataclass(frozen=True)
class ServerConfig:
    model_path: Path
    llama_cli_path: Path
    wrapper_path: Path
    timeout_seconds: float | None = 300.0

    @classmethod
    def from_env(cls) -> "ServerConfig":
        model_path = _resolve_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        llama_cli_path = _resolve_path(os.getenv("LLAMA_CLI_PATH", str(DEFAULT_LLAMACLI_PATH)))
        wrapper_path = _ensure_wrapper(llama_cli_path)
        timeout_seconds = _read_optional_timeout("LLAMA_CLI_TIMEOUT_SECONDS", 300.0)
        return cls(
            model_path=model_path,
            llama_cli_path=llama_cli_path,
            wrapper_path=wrapper_path,
            timeout_seconds=timeout_seconds,
        )

    def build_command(self, prompt: str) -> list[str]:
        n_predict = os.getenv("LLAMA_CLI_N_PREDICT", "256")
        threads   = os.getenv("LLAMA_CLI_THREADS", "4")
        ngl       = os.getenv("LLAMA_CLI_NGL", "99")
        return [
            str(self.wrapper_path),   # shell wrapper — sets LD_LIBRARY_PATH then execs llama-cli
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", n_predict,
            "-t", threads,
            "-ngl", ngl,
            "--simple-io",
            "--log-disable",
            "--no-display-prompt",
        ]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _read_optional_timeout(name: str, default: float) -> float | None:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        val = default
    else:
        try:
            val = float(raw)
        except ValueError as e:
            raise ValueError(f"{name} must be a number") from e
    return None if val <= 0 else val


def _shorten(text: str, limit: int = 2048) -> str:
    return text if len(text) <= limit else f"{text[:limit]}...[+{len(text)-limit}]"


def _run_llama_cli(command: list[str], timeout_seconds: float | None) -> subprocess.CompletedProcess[str]:
    # No env= needed — wrapper script handles LD_LIBRARY_PATH
    proc = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    logger.info("llama-cli pid=%s", proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(proc.args, timeout_seconds)

    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


# ── App factory ─────────────────────────────────────────────────────────────

def create_app(config: ServerConfig | None = None) -> FastAPI:
    runtime_config = config or ServerConfig.from_env()

    _tasks: dict[str, dict] = {}
    # max_workers=1 → one inference at a time (GPU can only run one llama-cli process)
    _executor = ThreadPoolExecutor(max_workers=1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup
        ok = True
        if not runtime_config.llama_cli_path.exists():
            logger.error("startup: llama-cli not found: %s", runtime_config.llama_cli_path)
            ok = False
        if not runtime_config.model_path.exists():
            logger.error("startup: model not found: %s", runtime_config.model_path)
            ok = False
        if not runtime_config.wrapper_path.exists():
            logger.error("startup: wrapper not found: %s", runtime_config.wrapper_path)
            ok = False
        app.state.ready = ok
        logger.info("startup done ready=%s wrapper=%s", ok, runtime_config.wrapper_path)
        yield
        # shutdown
        _executor.shutdown(wait=False)
        logger.info("executor shut down")

    app = FastAPI(
        title="HiSLM Inference Server — AGX Orin",
        version="1.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _run_inference_background(task_id: str, prompt: str) -> None:
        t0 = perf_counter()
        _tasks[task_id] = {"status": "running"}
        try:
            cmd = runtime_config.build_command(prompt)
            logger.info("task=%s inference start cmd=%s", task_id, " ".join(cmd))
            completed = _run_llama_cli(cmd, runtime_config.timeout_seconds)
            latency_ms = round((perf_counter() - t0) * 1000, 2)

            if completed.returncode != 0:
                err = (completed.stderr or "").strip()
                logger.error("task=%s failed rc=%s err=%s", task_id, completed.returncode, _shorten(err))
                _tasks[task_id] = {"status": "failed", "error": err or "inference failed", "latency_ms": latency_ms}
            else:
                text = (completed.stdout or "").strip()
                logger.info("task=%s done len=%d latency_ms=%.0f", task_id, len(text), latency_ms)
                _tasks[task_id] = {"status": "completed", "response": text, "latency_ms": latency_ms}

        except subprocess.TimeoutExpired:
            logger.warning("task=%s timed out", task_id)
            _tasks[task_id] = {"status": "failed", "error": "inference timed out"}
        except Exception as exc:
            logger.exception("task=%s unexpected error", task_id)
            _tasks[task_id] = {"status": "failed", "error": str(exc)}

    @app.post("/query")
    async def query(payload: QueryRequest) -> dict:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(422, "prompt must not be empty")
        if not getattr(app.state, "ready", False):
            raise HTTPException(503, "server not ready — check model/binary paths")

        logger.info("query prompt_chars=%d sensor=%s", len(prompt),
                    json.dumps(payload.sensor, ensure_ascii=True, default=str))

        task_id = str(uuid.uuid4())
        _tasks[task_id] = {"status": "queued"}
        asyncio.get_running_loop().run_in_executor(_executor, _run_inference_background, task_id, prompt)
        return {"status": "ok", "task_id": task_id}

    @app.get("/result/{task_id}")
    async def get_result(task_id: str) -> dict:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "task not found")
        return task

    @app.post("/ack/{task_id}")
    async def ack(task_id: str) -> dict:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "task not found")
        status = task.get("status", "unknown")
        logger.info("ack task=%s status=%s from %s", task_id, status, "client")
        return {"status": "acknowledged", "task_id": task_id, "task_status": status}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok" if getattr(app.state, "ready", False) else "not_ready"}

    @app.get("/status")
    async def status() -> dict:
        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
        for t in _tasks.values():
            s = t.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return {"ready": getattr(app.state, "ready", False), **counts}

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        reload=False,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
