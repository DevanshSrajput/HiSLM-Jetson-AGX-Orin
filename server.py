from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import datetime
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

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
        return [
            str(self.llama_cli_path),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
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

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest) -> QueryResponse:
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

        command = runtime_config.build_command(prompt)
        started_at = perf_counter()
        start_dt = datetime.datetime.utcnow()
        last_inference = getattr(app.state, "last_inference", None) or {}
        last_inference.update({"started_at": start_dt.isoformat(), "status": "running"})
        app.state.last_inference = last_inference

        logger.info("invoking llama-cli; cmd=%s", " ".join(map(str, command)))

        try:
            completed = await asyncio.to_thread(
                _run_llama_cli,
                command,
                runtime_config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            logger.warning(
                "llama-cli timed out after %.2fs",
                runtime_config.timeout_seconds,
            )
            raise HTTPException(status_code=504, detail="inference timed out") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail="llama-cli binary is not executable") from exc
        except OSError as exc:
            logger.exception("failed to launch llama-cli")
            raise HTTPException(status_code=500, detail="failed to launch inference process") from exc
        finally:
            # record end timestamp in any case
            end_dt = datetime.datetime.utcnow()
            last_inference = getattr(app.state, "last_inference", None) or {}
            last_inference.update({"ended_at": end_dt.isoformat()})
            app.state.last_inference = last_inference

        latency_ms = (perf_counter() - started_at) * 1000.0

        last_inference = getattr(app.state, "last_inference", None) or {}
        last_inference.update({"duration_ms": latency_ms})

        if completed.returncode != 0:
            stderr_text = (completed.stderr or "").strip()
            logger.error("llama-cli returncode=%s stderr_len=%d", completed.returncode, len(stderr_text))
            if stderr_text:
                logger.error("llama-cli failed: %s", _shorten(stderr_text))
            last_inference.update({"status": "failed", "stderr_len": len(stderr_text)})
            app.state.last_inference = last_inference
            raise HTTPException(status_code=502, detail="inference failed")

        response_text = (completed.stdout or "").rstrip("\r\n")
        logger.info(
            "request completed latency_ms=%.2f stdout_len=%d stderr_len=%d",
            latency_ms,
            len(response_text),
            len(completed.stderr or ""),
        )

        last_inference.update({"status": "ok", "response_len": len(response_text)})
        app.state.last_inference = last_inference

        return QueryResponse(response=response_text, latency_ms=latency_ms)

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