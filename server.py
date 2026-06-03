from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid
import datetime
from typing import Optional
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("llama_inference_server")


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    sensor: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    response: str
    latency_ms: float


@dataclass(frozen=True)
class ServerConfig:
    model_path: Path
    llama_cli_path: Path
    timeout_seconds: float | None = 300.0

    @classmethod
    def from_env(cls) -> "ServerConfig":
        model_path = _resolve_path(
            os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        llama_cli_path = _resolve_path(
            os.getenv("LLAMA_CLI_PATH", str(DEFAULT_LLAMACLI_PATH))
        )
        timeout_seconds = _read_optional_timeout("LLAMA_CLI_TIMEOUT_SECONDS", 300.0)
        return cls(
            model_path=model_path,
            llama_cli_path=llama_cli_path,
            timeout_seconds=timeout_seconds,
        )

    def build_command(self, prompt: str) -> list[str]:
        n_predict = os.getenv("LLAMA_CLI_N_PREDICT", "256")
        threads = os.getenv("LLAMA_CLI_THREADS", "4")
        return [
            str(self.llama_cli_path),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n", n_predict,
            "-t", threads,
            "--simple-io",
            "--log-disable",
        ]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _read_optional_timeout(name: str, default: float) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        timeout_value = default
    else:
        try:
            timeout_value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc

    if timeout_value <= 0:
        return None
    return timeout_value


def _shorten(text: str, limit: int = 4096) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _run_llama_cli(command: list[str], timeout_seconds: float | None) -> subprocess.CompletedProcess[str]:
    """
    Run llama-cli using Popen so we can log PID and handle timeouts robustly.
    Returns a CompletedProcess-like object with stdout/stderr populated.
    """
    proc = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    logger.info("llama-cli started pid=%s", getattr(proc, "pid", "?"))
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        try:
            proc.kill()
        except Exception:
            logger.exception("failed to kill timed-out llama-cli process")
        # collect whatever output is available
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(proc.args, timeout_seconds)

    completed = subprocess.CompletedProcess(proc.args, proc.returncode, stdout=stdout, stderr=stderr)
    return completed


def create_app(config: ServerConfig | None = None) -> FastAPI:
    runtime_config = config or ServerConfig.from_env()

    app = FastAPI(
        title="Jetson Llama Inference Server",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _tasks: dict[str, dict] = {}
    _executor = ThreadPoolExecutor(max_workers=1)

    def _run_inference_background(task_id: str, prompt: str) -> None:
        _tasks[task_id] = {"status": "running"}
        try:
            command = runtime_config.build_command(prompt)
            logger.info("inference task=%s starting llama-cli", task_id)
            completed = _run_llama_cli(command, runtime_config.timeout_seconds)
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                logger.error("inference task=%s failed returncode=%s", task_id, completed.returncode)
                _tasks[task_id] = {"status": "failed", "error": stderr or "inference failed"}
            else:
                response_text = (completed.stdout or "").rstrip("\r\n")
                _tasks[task_id] = {"status": "completed", "response": response_text}
                logger.info("inference task=%s completed len=%d", task_id, len(response_text))
        except subprocess.TimeoutExpired:
            logger.warning("inference task=%s timed out", task_id)
            _tasks[task_id] = {"status": "failed", "error": "inference timed out"}
        except Exception as exc:
            logger.exception("inference task=%s failed unexpectedly", task_id)
            _tasks[task_id] = {"status": "failed", "error": str(exc)}

    @app.post("/query")
    async def query(payload: QueryRequest) -> dict:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="prompt must not be empty")

        logger.info(
            "request received prompt_chars=%d sensor=%s",
            len(prompt),
            json.dumps(payload.sensor, ensure_ascii=True, default=str),
        )

        if not runtime_config.llama_cli_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"llama-cli binary not found: {runtime_config.llama_cli_path}",
            )

        if not runtime_config.model_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"model file not found: {runtime_config.model_path}",
            )

        task_id = str(uuid.uuid4())
        _tasks[task_id] = {"status": "queued"}

        asyncio.get_running_loop().run_in_executor(_executor, _run_inference_background, task_id, prompt)

        return {"status": "ok", "task_id": task_id}

    @app.get("/result/{task_id}")
    async def get_result(task_id: str) -> dict:
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.on_event("startup")
    async def _startup_checks() -> None:
        cfg = ServerConfig.from_env()
        ok = True
        if not cfg.llama_cli_path.exists():
            logger.error("startup check: llama-cli not found: %s", cfg.llama_cli_path)
            ok = False
        if not cfg.model_path.exists():
            logger.error("startup check: model not found: %s", cfg.model_path)
            ok = False

        app.state.ready = ok
        app.state.last_inference = {"status": "idle"}


    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok" if getattr(app.state, "ready", False) else "not_ready"}


    @app.get("/status")
    async def status() -> dict:
        return {"ready": getattr(app.state, "ready", False), "last_inference": getattr(app.state, "last_inference", {})}

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
