"""
server2.py — Run on AGX Orin (Tailscale / wireless setup)
===========================================================
SLM inference server for wireless communication with Orin NX
over Tailscale VPN. Auto-detects Tailscale IP and serves
WebSocket + REST API for chat / inference.

Endpoints:
  - WebSocket   /ws
  - REST        POST /send, GET /messages
  - UI          GET /  → index.html (main), GET /nx → nx_index.html (NX client)
  - Health      GET /health

Usage:
  python server2.py

On NX:
  python client2.py --agx-ip <TAILSCALE_IP>
  # or open http://<TAILSCALE_IP>:8000 in browser
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

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("server2.log", mode="a"),
    ],
)
log = logging.getLogger("AGX-SERVER2")

HOST = "0.0.0.0"
PORT = 8000
MAX_HISTORY = 200
PING_INTERVAL = 20
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "llama.cpp" / "models" / "Qwen2.5-3B-Q4_K_M.gguf"
DEFAULT_LLAMA_RUNNER = BASE_DIR / "llama.cpp" / "build" / "bin" / "llama-cli"

LLAMA_SYSTEM_PROMPT = (
    "You are HiSLM running on the AGX Orin. Reply directly to the NX client's "
    "question in a concise, useful way. Do not include hidden reasoning, logs, "
    "or prompt text."
)

def get_tailscale_ip() -> str:
    try:
        result = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            ip = result.stdout.strip()
            if ip:
                return ip
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    try:
        result = subprocess.run(['ip', 'addr'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            parts = line.strip().split()
            for i, part in enumerate(parts):
                if part.startswith('100.') and '/' in part:
                    return part.split('/')[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return ''

TAILSCALE_IP = get_tailscale_ip()

class Message(BaseModel):
    id: str
    sender: str
    role: str
    text: str
    timestamp: str

class SendRequest(BaseModel):
    sender: str
    text: str

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

@dataclass(frozen=True)
class LlamaConfig:
    runner_path: Path
    model_path: Path
    timeout_seconds: float | None = 300.0
    n_predict: str = "256"
    threads: str = "8"
    ctx_size: str = "8192"
    gpu_layers: str = "99"

    @classmethod
    def from_env(cls) -> "LlamaConfig":
        return cls(
            runner_path=_resolve_path(os.getenv("LLAMA_RUNNER", str(DEFAULT_LLAMA_RUNNER))),
            model_path=_resolve_path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            timeout_seconds=_read_optional_timeout("LLAMA_TIMEOUT_SECONDS", 300.0),
            n_predict=os.getenv("LLAMA_N_PREDICT", "256"),
            threads=os.getenv("LLAMA_THREADS", "8"),
            ctx_size=os.getenv("LLAMA_CTX_SIZE", "8192"),
            gpu_layers=os.getenv("LLAMA_GPU_LAYERS", "99"),
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
            "--flash-attn", "auto",
            "--no-warmup",
            "--single-turn",
            "--simple-io",
            "-e",
            "--no-display-prompt",
            "--no-show-timings",
            "--log-disable",
            "--repeat-penalty", "1.15",
            "--temperature", "0.7",
            "--top-k", "40",
            "--top-p", "0.95",
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
    lines = stdout.splitlines()
    out = []
    capture = False
    for line in lines:
        if capture:
            if line.strip() == "Exiting...":
                break
            out.append(line)
        elif line.startswith("> "):
            capture = True
    while out and not out[0].strip():
        out.pop(0)
    reply = "\n".join(out).strip()
    if not reply and "<|im_start|>assistant" in stdout:
        idx = stdout.index("<|im_start|>assistant") + len("<|im_start|>assistant")
        after = stdout[idx:].strip()
        if "<|im_end|>" in after:
            after = after[:after.index("<|im_end|>")]
        reply = after.strip()
    return reply

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

class ConnectionManager:
    def __init__(self):
        self.active: dict[str, WebSocket] = {}

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

app = FastAPI(title="HiSLM AGX Server — Tailscale Wireless", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    log.info(f"[STATIC] Serving files from {STATIC_DIR}")

@app.get("/health")
async def health():
    log.debug("[HEALTH] ping received")
    llama_errors = llama_config.validate()
    return {
        "status": "ok",
        "node": "AGX-Orin-30GB",
        "tailscale_ip": TAILSCALE_IP or "not detected",
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
    log.info(f"[GET /messages] limit={limit}")
    return {"messages": message_history[-limit:]}

@app.post("/send")
async def send_message(req: SendRequest):
    log.info(f"[POST /send] sender={req.sender!r}  text={req.text[:80]!r}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    role = "server" if req.sender.lower() in ("agx", "server", "agx-orin") else "user"
    msg = make_message(sender=req.sender, role=role, text=req.text.strip())
    await manager.broadcast({"type": "message", "payload": msg})
    if role != "server":
        asyncio.create_task(generate_and_broadcast_reply(msg["text"]))
    log.info(f"[POST /send] broadcast done  id={msg['id']}")
    return {"ok": True, "message": msg}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, client_id: str = "anon"):
    await manager.connect(client_id, ws)
    history_payload = {"type": "history", "payload": message_history[-MAX_HISTORY:]}
    await ws.send_text(json.dumps(history_payload))
    log.info(f"[WS] Sent history ({len(message_history)} msgs) → {client_id}")
    await ws.send_text(json.dumps({
        "type": "connected",
        "client_id": client_id,
        "node": "AGX-Orin-30GB",
        "tailscale_ip": TAILSCALE_IP,
    }))
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
                if role != "server":
                    asyncio.create_task(generate_and_broadcast_reply(msg["text"]))
                try:
                    await ws.send_text(json.dumps({"type": "ack", "id": msg["id"]}))
                except Exception:
                    log.warning(f"[WS ACK FAIL] {client_id}: ack send failed, continuing")
                await manager.broadcast(
                    {"type": "message", "payload": msg},
                    exclude=client_id,
                )
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
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        log.debug("[UI] Serving index.html")
        return HTMLResponse(content=ui_path.read_text())
    log.debug("[UI] Serving inline fallback HTML")
    return HTMLResponse(content=INLINE_HTML)

@app.get("/nx", response_class=HTMLResponse)
async def serve_nx_ui():
    nx_path = Path(__file__).parent / "static" / "nx_index.html"
    if nx_path.exists():
        log.debug("[UI] Serving nx_index.html for NX client")
        return HTMLResponse(content=nx_path.read_text())
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html><head><title>HiSLM NX Client</title></head>
<body style="font-family:monospace;padding:2rem;background:#0d0d0d;color:#e0e0e0">
<h2>HiSLM NX Wireless Client</h2>
<p>Place <strong>static/nx_index.html</strong> next to server2.py for the NX GUI.</p>
<p>Tailscale IP: <strong>{TAILSCALE_IP or 'not detected'}</strong></p>
</body></html>""")

INLINE_HTML = """<!DOCTYPE html>
<html><head><title>HiSLM Node Chat</title></head>
<body style="font-family:monospace;padding:2rem;background:#0d0d0d;color:#e0e0e0">
<h2>HiSLM AGX Server — Tailscale Wireless</h2>
<p>Place <strong>static/index.html</strong> next to server2.py for the full UI.</p>
<p>Server is running on <strong>port 8000</strong>.</p>
<p>Tailscale IP: <strong>{TAILSCALE_IP or 'not detected'}</strong></p>
<p><a href="/health" style="color:#4af">Health check</a></p>
</body></html>"""

def _open_browser_when_ready(port: int, delay: float = 1.5):
    time.sleep(delay)
    loc_url = f"http://localhost:{port}/?client_id=agx-operator&role=agx"
    ts_url = f"http://{TAILSCALE_IP}:{port}/?client_id=agx-operator&role=agx" if TAILSCALE_IP else None
    url = loc_url
    log.info(f"[BROWSER] Opening UI → {url}")
    if ts_url:
        log.info(f"[BROWSER] NX client → {ts_url}")
    try:
        webbrowser.open(url)
    except Exception as exc:
        log.warning(f"[BROWSER] Could not open browser automatically: {exc}")
        log.info(f"[BROWSER] Open manually: {url}")

if __name__ == "__main__":
    lan_ip = "127.0.0.1"
    try:
        with __import__('socket').socket(__import__('socket').AF_INET, __import__('socket').SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
    except Exception:
        pass

    log.info("=" * 60)
    log.info("  HiSLM AGX Server — Tailscale Wireless")
    log.info(f"  LAN IP      : http://{lan_ip}:{PORT}")
    if TAILSCALE_IP:
        log.info(f"  TAILSCALE   : http://{TAILSCALE_IP}:{PORT}  ← use this from NX")
    else:
        log.info(f"  TAILSCALE   : not detected (install Tailscale or set manually)")
    log.info(f"  NX GUI      : http://{lan_ip}:{PORT}/nx")
    log.info(f"  Inference   : Qwen2.5-3B (llama.cpp)")
    log.info("=" * 60)

    print(f"\n{'='*60}")
    print(f"  HiSLM AGX Server")
    print(f"  LAN   → http://{lan_ip}:{PORT}")
    if TAILSCALE_IP:
        print(f"  WIRE   → http://{TAILSCALE_IP}:{PORT}  (Tailscale)")
    print(f"  NX UI → http://{lan_ip}:{PORT}/nx")
    print(f"{'='*60}\n")

    if TAILSCALE_IP:
        print(f"  On NX: python client2.py --agx-ip {TAILSCALE_IP}")
        print()

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
