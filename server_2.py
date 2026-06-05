"""
server_2.py — Run on any node (configurable network)
======================================================
Same as server.py but HOST / PORT / node-name are configurable
via CLI arguments or environment variables, so this can be
deployed on a different machine / network without editing the file.

Usage:
  python server_2.py                          # defaults: 0.0.0.0:8001
  python server_2.py --host 192.168.1.10 --port 9000 --node-name MY-SERVER
  SERVER_HOST=10.0.0.5 SERVER_PORT=9000 python server_2.py

Then open http://<HOST>:<PORT> in a browser.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
import webbrowser
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
# CLI args / env config  (only new addition vs server.py)
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HiSLM Node Messenger — configurable server")
    p.add_argument("--host",      default=os.getenv("SERVER_HOST", "0.0.0.0"),
                   help="Bind address (default: 0.0.0.0 or $SERVER_HOST)")
    p.add_argument("--port",      type=int, default=int(os.getenv("SERVER_PORT", "8001")),
                   help="Bind port (default: 8001 or $SERVER_PORT)")
    p.add_argument("--node-name", default=os.getenv("SERVER_NODE_NAME", "Server-Node-2"),
                   help="Node display name (default: Server-Node-2 or $SERVER_NODE_NAME)")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not auto-open browser on start")
    return p.parse_args()

_args = parse_args()

HOST      = _args.host
PORT      = _args.port
NODE_NAME = _args.node_name

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("server_2.log", mode="a"),
    ],
)
log = logging.getLogger(f"{NODE_NAME}-SERVER")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MAX_HISTORY    = 200
PING_INTERVAL  = 20

# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────
class Message(BaseModel):
    id: str
    sender: str
    role: str
    text: str
    timestamp: str

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
# WebSocket connection manager
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="HiSLM Node Messenger 2", version="1.0.0")

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

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    log.debug("[HEALTH] ping received")
    return {
        "status": "ok",
        "node": NODE_NAME,
        "connected_clients": len(manager.active),
        "message_count": len(message_history),
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

    role = "server" if req.sender.lower() in ("agx", "server", "agx-orin", NODE_NAME.lower()) else "user"
    msg = make_message(sender=req.sender, role=role, text=req.text.strip())

    await manager.broadcast({"type": "message", "payload": msg})
    log.info(f"[POST /send] broadcast done  id={msg['id']}")
    return {"ok": True, "message": msg}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, client_id: str = "anon"):
    """
    WebSocket endpoint.
    Query param: ws://host:<PORT>/ws?client_id=<name>

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

    history_payload = {"type": "history", "payload": message_history[-MAX_HISTORY:]}
    await ws.send_text(json.dumps(history_payload))
    log.info(f"[WS] Sent history ({len(message_history)} msgs) → {client_id}")

    await ws.send_text(json.dumps({
        "type": "connected",
        "client_id": client_id,
        "node": NODE_NAME,
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

                role = "server" if sender.lower() in ("agx", "server", "agx-orin", NODE_NAME.lower()) else "user"
                msg = make_message(sender=sender, role=role, text=text)

                await ws.send_text(json.dumps({"type": "ack", "id": msg["id"]}))
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
        log.debug("[UI] Serving from file")
        return HTMLResponse(content=ui_path.read_text())
    log.debug("[UI] Serving inline fallback HTML")
    return HTMLResponse(content=INLINE_HTML)


# ─────────────────────────────────────────────
# Inline fallback UI
# ─────────────────────────────────────────────
INLINE_HTML = f"""<!DOCTYPE html>
<html><head><title>HiSLM Node Chat — {NODE_NAME}</title></head>
<body style="font-family:monospace;padding:2rem;background:#0d0d0d;color:#e0e0e0">
<h2>HiSLM Node Messenger — {NODE_NAME}</h2>
<p>Place <strong>static/index.html</strong> next to server_2.py for the full UI.</p>
<p>Server is running on <strong>port {PORT}</strong>.</p>
<p><a href="/health" style="color:#4af">Health check</a></p>
</body></html>"""


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def _open_browser_when_ready(port: int, delay: float = 1.5):
    time.sleep(delay)
    url = f"http://localhost:{port}/?client_id=operator&role=server"
    log.info(f"[BROWSER] Opening UI → {url}")
    try:
        webbrowser.open(url)
    except Exception as exc:
        log.warning(f"[BROWSER] Could not open browser automatically: {exc}")
        log.info(f"[BROWSER] Open manually: {url}")


if __name__ == "__main__":
    import socket as _socket

    # Resolve actual LAN IP when bound to 0.0.0.0
    if HOST in ("0.0.0.0", ""):
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as _s:
                _s.connect(("8.8.8.8", 80))
                _actual_ip = _s.getsockname()[0]
        except Exception:
            _actual_ip = "127.0.0.1"
    else:
        _actual_ip = HOST

    log.info("=" * 60)
    log.info(f"  HiSLM Node Messenger — {NODE_NAME}")
    log.info(f"  Bind address : {HOST}:{PORT}")
    log.info(f"  LAN access   : http://{_actual_ip}:{PORT}")
    log.info(f"  Local access : http://localhost:{PORT}/")
    log.info("=" * 60)

    print("\n" + "=" * 60)
    print(f"  HiSLM Node Messenger — {NODE_NAME}")
    print(f"  IP   : {_actual_ip}")
    print(f"  Port : {PORT}")
    print(f"  URL  : http://{_actual_ip}:{PORT}")
    print("=" * 60 + "\n")

    if not _args.no_browser:
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
