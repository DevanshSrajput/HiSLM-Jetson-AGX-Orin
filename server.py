"""
server.py — Run on AGX Orin (30GB)
====================================
Parent node. Hosts:
  - WebSocket endpoint  (/ws)
  - REST fallback       (POST /send, GET /messages)
  - Static UI           (GET / → index.html)
  - Health check        (GET /health)

Start:
  python server.py

Then open http://<AGX_IP>:8000 in a browser on either device.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("server.log", mode="a"),
    ],
)
log = logging.getLogger("AGX-SERVER")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
MAX_HISTORY = 200          # messages kept in memory
PING_INTERVAL = 20         # WebSocket keepalive ping interval (seconds)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
DEFAULT_LLAMA_RUNNER = BASE_DIR / "run_llama.sh"

LLAMA_SYSTEM_PROMPT = (
    "You are HiSLM running on the AGX Orin. Reply directly to the NX client's "
    "question in a concise, useful way. Do not include hidden reasoning, logs, "
    "or prompt text."
)

# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────
class Message(BaseModel):
    id: str
    sender: str            # "agx" | "nx" | device hostname
    role: str              # "user" | "server"  (server = AGX operator)
    text: str
    timestamp: str         # ISO-8601 UTC

class SendRequest(BaseModel):
    sender: str
    text: str

# ─────────────────────────────────────────────
# In-memory message store
# ─────────────────────────────────────────────
message_history: List[dict] = []

def make_message(sender: str, role: str, text: str) -> dict:
    msg = {
        "id": str(uuid.uuid4()),
        "sender": sender,
        "role": role,
        "text": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    message_history.append(msg)
    if len(message_history) > MAX_HISTORY:
        message_history.pop(0)
    log.info(f"[MSG STORED] sender={sender} role={role} text={text[:80]!r}")
    return msg

# ─────────────────────────────────────────────
# llama.cpp / Qwen responder
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class LlamaConfig:
    runner_path: Path
    model_path: Path
    timeout_seconds: float | None = 300.0
    n_predict: str = "256"
    threads: str = "4"
    ctx_size: str = "4096"
    gpu_layers: str = "auto"

    @classmethod
    def from_env(cls) -> "LlamaConfig":
        return cls(
            runner_path=_resolve_path(os.getenv("LLAMA_RUNNER", str(DEFAULT_LLAMA_RUNNER))),
            model_path=_resolve_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            timeout_seconds=_read_optional_timeout("LLAMA_TIMEOUT_SECONDS", 300.0),
            n_predict=os.getenv("LLAMA_N_PREDICT", "256"),
            threads=os.getenv("LLAMA_THREADS", "4"),
            ctx_size=os.getenv("LLAMA_CTX_SIZE", "4096"),
            gpu_layers=os.getenv("LLAMA_GPU_LAYERS", "auto"),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.runner_path.exists():
            errors.append(f"llama runner not found: {self.runner_path}")
        if not self.model_path.exists():
            errors.append(f"Qwen model not found: {self.model_path}")
        return errors

    def build_command(self, prompt: str) -> list[str]:
        return [
            str(self.runner_path),
            "-m", str(self.model_path),
            "-sys", LLAMA_SYSTEM_PROMPT,
            "-p", prompt,
            "-n", self.n_predict,
            "-t", self.threads,
            "-c", self.ctx_size,
            "-ngl", self.gpu_layers,
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--no-show-timings",
            "--log-disable",
        ]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _read_optional_timeout(name: str, default: float) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            log.warning("[%s] invalid value %r, using %.1fs", name, raw, default)
            value = default
    return None if value <= 0 else value


llama_config = LlamaConfig.from_env()
llama_executor = ThreadPoolExecutor(max_workers=1)


def run_llama_reply(question: str) -> str:
    errors = llama_config.validate()
    if errors:
        raise RuntimeError("; ".join(errors))

    command = llama_config.build_command(question)
    log.info("[LLAMA] starting Qwen2.5-3B response prompt_chars=%d", len(question))
    completed = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=llama_config.timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(stderr or f"llama-cli exited with code {completed.returncode}")

    reply = clean_llama_stdout(completed.stdout or "")
    if not reply:
        raise RuntimeError("llama-cli returned an empty response")
    log.info("[LLAMA] completed response_chars=%d", len(reply))
    return reply


def clean_llama_stdout(stdout: str) -> str:
    lines = stdout.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    answer_lines: list[str] = []
    saw_prompt = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("> "):
            saw_prompt = True
            answer_lines.clear()
            continue
        if not saw_prompt:
            continue
        if stripped == "Exiting...":
            break
        answer_lines.append(line)

    reply = "\n".join(answer_lines).strip()
    return reply or stdout.strip()


async def generate_and_broadcast_reply(question: str):
    loop = asyncio.get_running_loop()
    try:
        reply = await loop.run_in_executor(llama_executor, run_llama_reply, question)
    except subprocess.TimeoutExpired:
        log.warning("[LLAMA] inference timed out")
        reply = "Qwen inference timed out before I could finish the reply."
    except Exception as exc:
        log.error("[LLAMA] inference failed: %s", exc, exc_info=True)
        reply = f"Qwen inference failed on AGX: {exc}"

    msg = make_message(sender="agx-qwen2.5-3b", role="server", text=reply)
    await manager.broadcast({"type": "message", "payload": msg})

# ─────────────────────────────────────────────
# WebSocket connection manager
# ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}   # client_id → websocket

    async def connect(self, client_id: str, ws: WebSocket):
        await ws.accept()
        self.active[client_id] = ws
        log.info(f"[WS CONNECT] client_id={client_id}  total_connected={len(self.active)}")

    def disconnect(self, client_id: str):
        ws = self.active.pop(client_id, None)
        log.info(f"[WS DISCONNECT] client_id={client_id}  remaining={len(self.active)}")
        return ws

    async def send_to(self, client_id: str, payload: dict) -> bool:
        ws = self.active.get(client_id)
        if ws is None:
            log.warning(f"[WS SEND FAIL] client_id={client_id} not connected")
            return False
        try:
            await ws.send_text(json.dumps(payload))
            log.debug(f"[WS SEND OK] → {client_id}  payload_type={payload.get('type')}")
            return True
        except Exception as exc:
            log.error(f"[WS SEND ERROR] → {client_id}: {exc}")
            self.active.pop(client_id, None)
            return False

    async def broadcast(self, payload: dict, exclude: Optional[str] = None):
        targets = [cid for cid in list(self.active.keys()) if cid != exclude]
        log.debug(f"[WS BROADCAST] payload_type={payload.get('type')}  targets={targets}")
        for cid in targets:
            await self.send_to(cid, payload)

manager = ConnectionManager()

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="HiSLM Node Messenger", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI) if the folder exists
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    log.info(f"[STATIC] Serving files from {STATIC_DIR}")

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    log.debug("[HEALTH] ping received")
    llama_errors = llama_config.validate()
    return {
        "status": "ok",
        "node": "AGX-Orin-30GB",
        "connected_clients": len(manager.active),
        "message_count": len(message_history),
        "llama_ready": not llama_errors,
        "llama_model": str(llama_config.model_path),
        "llama_runner": str(llama_config.runner_path),
        "llama_errors": llama_errors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/messages")
async def get_messages(limit: int = 50):
    """REST: fetch last N messages (for clients that can't use WS)."""
    log.info(f"[GET /messages] limit={limit}")
    return {"messages": message_history[-limit:]}


@app.post("/send")
async def send_message(req: SendRequest):
    """
    REST: post a message.
    The message is stored and broadcast to all connected WS clients.
    """
    log.info(f"[POST /send] sender={req.sender!r}  text={req.text[:80]!r}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    role = "server" if req.sender.lower() in ("agx", "server", "agx-orin") else "user"
    msg = make_message(sender=req.sender, role=role, text=req.text.strip())

    # Push to all WebSocket clients
    await manager.broadcast({"type": "message", "payload": msg})
    if role != "server":
        asyncio.create_task(generate_and_broadcast_reply(msg["text"]))
    log.info(f"[POST /send] broadcast done  id={msg['id']}")
    return {"ok": True, "message": msg}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, client_id: str = "anon"):
    """
    WebSocket endpoint.
    Query param: ws://host:8000/ws?client_id=nx-node-1
    
    Protocol (JSON frames):
      Client → Server:
        { "type": "message",  "sender": "nx", "text": "hello" }
        { "type": "ping" }
      
      Server → Client:
        { "type": "message",  "payload": <Message dict> }
        { "type": "history",  "payload": [<Message>, ...] }
        { "type": "pong" }
        { "type": "ack",      "id": "..." }
        { "type": "connected","client_id": "..." }
    """
    await manager.connect(client_id, ws)

    # Send message history on connect
    history_payload = {"type": "history", "payload": message_history[-MAX_HISTORY:]}
    await ws.send_text(json.dumps(history_payload))
    log.info(f"[WS] Sent history ({len(message_history)} msgs) → {client_id}")

    # Notify client of successful connection
    await ws.send_text(json.dumps({
        "type": "connected",
        "client_id": client_id,
        "node": "AGX-Orin-30GB",
    }))

    # Notify all others that someone joined
    system_msg = make_message(
        sender="system",
        role="system",
        text=f"[{client_id}] connected",
    )
    await manager.broadcast({"type": "message", "payload": system_msg}, exclude=client_id)

    try:
        while True:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=PING_INTERVAL + 5)
            log.debug(f"[WS RECV] from={client_id}  raw={raw[:120]!r}")

            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(f"[WS BAD JSON] from={client_id}")
                await ws.send_text(json.dumps({"type": "error", "detail": "invalid JSON"}))
                continue

            ftype = frame.get("type", "")

            if ftype == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
                log.debug(f"[WS PING/PONG] {client_id}")

            elif ftype == "message":
                text = (frame.get("text") or "").strip()
                sender = frame.get("sender") or client_id
                if not text:
                    log.warning(f"[WS MSG] empty text from {client_id}, ignoring")
                    continue

                role = "server" if sender.lower() in ("agx", "server", "agx-orin") else "user"
                msg = make_message(sender=sender, role=role, text=text)

                # Ack back to sender
                await ws.send_text(json.dumps({"type": "ack", "id": msg["id"]}))
                # Broadcast to everyone else
                await manager.broadcast(
                    {"type": "message", "payload": msg},
                    exclude=client_id,
                )
                if role != "server":
                    asyncio.create_task(generate_and_broadcast_reply(msg["text"]))
                log.info(f"[WS MSG] stored + broadcast  id={msg['id']}")

            else:
                log.warning(f"[WS UNKNOWN] type={ftype!r} from={client_id}")

    except asyncio.TimeoutError:
        log.warning(f"[WS TIMEOUT] no frame from {client_id} in {PING_INTERVAL+5}s → closing")
    except WebSocketDisconnect as exc:
        log.info(f"[WS DISCONNECT] {client_id} code={exc.code}")
    except Exception as exc:
        log.error(f"[WS ERROR] {client_id}: {exc}", exc_info=True)
    finally:
        manager.disconnect(client_id)
        leave_msg = make_message(
            sender="system",
            role="system",
            text=f"[{client_id}] disconnected",
        )
        await manager.broadcast({"type": "message", "payload": leave_msg})
        log.info(f"[WS CLEANUP] {client_id} fully removed")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat UI (inline so no separate static dir is needed)."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        log.debug("[UI] Serving from file")
        return HTMLResponse(content=ui_path.read_text())
    log.debug("[UI] Serving inline fallback HTML")
    return HTMLResponse(content=INLINE_HTML)


# ─────────────────────────────────────────────
# Inline fallback UI (used if static/index.html absent)
# ─────────────────────────────────────────────
INLINE_HTML = """<!DOCTYPE html>
<html><head><title>HiSLM Node Chat</title></head>
<body style="font-family:monospace;padding:2rem;background:#0d0d0d;color:#e0e0e0">
<h2>HiSLM Node Messenger</h2>
<p>Place <strong>static/index.html</strong> next to server.py for the full UI.</p>
<p>Server is running on <strong>port 8000</strong>.</p>
<p><a href="/health" style="color:#4af">Health check</a></p>
</body></html>"""


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def _open_browser_when_ready(port: int, delay: float = 1.5):
    """
    Wait briefly for uvicorn to bind, then open the UI in the local browser.
    Runs in a daemon thread so it doesn't block the server.
    """
    time.sleep(delay)
    url = f"http://localhost:{port}/?client_id=agx-operator&role=agx"
    log.info(f"[BROWSER] Opening UI → {url}")
    try:
        webbrowser.open(url)
    except Exception as exc:
        log.warning(f"[BROWSER] Could not open browser automatically: {exc}")
        log.info(f"[BROWSER] Open manually: {url}")


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  HiSLM Node Messenger — AGX Orin (Server)")
    log.info(f"  Listening on http://{HOST}:{PORT}")
    log.info(f"  UI will auto-open at http://localhost:{PORT}/")
    log.info("=" * 60)

    # Open the browser in the background after uvicorn is up
    t = threading.Thread(
        target=_open_browser_when_ready,
        args=(PORT,),
        daemon=True,
    )
    t.start()

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        ws_ping_interval=PING_INTERVAL,
        ws_ping_timeout=30,
    )
