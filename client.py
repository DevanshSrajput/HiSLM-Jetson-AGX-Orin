"""
client.py — Run on Jetson Orin NX (8GB)
=========================================
Child node. Connects to AGX server over LAN.

Two modes:
  1. GUI mode (default):  opens the web UI in the browser
  2. CLI mode (--cli):    terminal chat, no browser required

Usage:
  # GUI (open browser on NX):
  python client.py --agx-ip 172.16.6.21

  # Headless / terminal mode:
  python client.py --agx-ip 172.16.6.21 --cli --name nx-node-1

  # Custom port:
  python client.py --agx-ip 172.16.6.21 --port 8001

The GUI is served from the AGX itself — client.py in GUI mode just
opens http://<AGX_IP>:<PORT>/?client_id=<name> in the local browser.
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone

import requests
import websocket   # websocket-client (sync)

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("client.log", mode="a"),
    ],
)
log = logging.getLogger("NX-CLIENT")

# ─────────────────────────────────────────────
# REST helpers (fallback / health check)
# ─────────────────────────────────────────────

def check_health(base_url: str, timeout: int = 5) -> bool:
    url = f"{base_url}/health"
    log.info(f"[HEALTH CHECK] GET {url}")
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        log.info(f"[HEALTH OK] {data}")
        return True
    except requests.RequestException as exc:
        log.error(f"[HEALTH FAIL] {exc}")
        return False


def rest_send(base_url: str, sender: str, text: str) -> dict:
    """Send via REST POST /send (fallback if WS unavailable)."""
    url = f"{base_url}/send"
    payload = {"sender": sender, "text": text}
    log.info(f"[REST SEND] POST {url}  payload={payload}")
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    data = r.json()
    log.info(f"[REST SEND OK] {data}")
    return data


def rest_get_messages(base_url: str, limit: int = 50) -> list:
    url = f"{base_url}/messages?limit={limit}"
    log.info(f"[REST GET] {url}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json().get("messages", [])


# ─────────────────────────────────────────────
# WebSocket CLI client
# ─────────────────────────────────────────────

class WSCliClient:
    """
    Synchronous WebSocket client for terminal use.
    Receives messages in a background thread, sends from main thread.
    """

    def __init__(self, ws_url: str, client_name: str):
        self.ws_url = ws_url
        self.name = client_name
        self._ws: websocket.WebSocketApp | None = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._recv_thread: threading.Thread | None = None

    # ── callbacks ─────────────────────────────

    def _on_open(self, ws):
        log.info(f"[WS OPEN] connected to {self.ws_url}")
        self._connected.set()

    def _on_message(self, ws, raw: str):
        log.debug(f"[WS RECV RAW] {raw[:200]!r}")
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("[WS RECV] invalid JSON, skipping")
            return

        ftype = frame.get("type", "")

        if ftype == "connected":
            print(f"\n✓ Connected as [{frame.get('client_id')}] on {frame.get('node')}\n")

        elif ftype == "history":
            msgs = frame.get("payload", [])
            log.info(f"[WS HISTORY] received {len(msgs)} messages")
            if msgs:
                print("\n── Message History ──")
                for m in msgs[-20:]:   # show last 20
                    self._print_msg(m)
                print("── End History ──\n")

        elif ftype == "message":
            msg = frame.get("payload", {})
            self._print_msg(msg)

        elif ftype == "ack":
            log.debug(f"[WS ACK] id={frame.get('id')}")

        elif ftype == "pong":
            log.debug("[WS PONG]")

        elif ftype == "error":
            log.warning(f"[WS SERVER ERROR] {frame.get('detail')}")
            print(f"\n⚠ Server error: {frame.get('detail')}\n")

    def _on_error(self, ws, error):
        log.error(f"[WS ERROR] {error}")
        print(f"\n⚠ WebSocket error: {error}\n")

    def _on_close(self, ws, code, msg):
        log.info(f"[WS CLOSE] code={code} msg={msg}")
        self._connected.clear()
        self._stop.set()
        print("\n[disconnected from server]\n")

    # ── helpers ───────────────────────────────

    @staticmethod
    def _print_msg(msg: dict):
        if msg.get("role") == "system":
            print(f"  ·  {msg.get('text', '')}")
            return
        ts = msg.get("timestamp", "")[:19].replace("T", " ")
        sender = msg.get("sender", "?")
        text = msg.get("text", "")
        role_tag = "[AGX]" if msg.get("role") == "server" else "[NX] "
        print(f"  {ts}  {role_tag}  {sender}: {text}")

    def _send_frame(self, frame: dict):
        if self._ws and self._connected.is_set():
            raw = json.dumps(frame)
            self._ws.send(raw)
            log.debug(f"[WS SENT] {raw[:120]!r}")
        else:
            log.warning("[WS SEND] not connected, dropping message")

    # ── lifecycle ─────────────────────────────

    def _run_ws(self):
        self._ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        log.info(f"[WS] Starting connection to {self.ws_url}")
        self._ws.run_forever(
            ping_interval=20,
            ping_timeout=10,
        )

    def connect(self, timeout: int = 10) -> bool:
        self._recv_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._recv_thread.start()
        ok = self._connected.wait(timeout=timeout)
        if not ok:
            log.error(f"[WS] Connection timed out after {timeout}s")
        return ok

    def send(self, text: str):
        frame = {
            "type": "message",
            "sender": self.name,
            "text": text,
        }
        self._send_frame(frame)
        log.info(f"[SEND] sender={self.name!r}  text={text[:80]!r}")

    def ping(self):
        self._send_frame({"type": "ping"})

    def close(self):
        self._stop.set()
        if self._ws:
            self._ws.close()
        log.info("[WS] Closed by client")

    def run_cli(self):
        """Blocking REPL — type messages, Ctrl-C to quit."""
        print("\n─────────────────────────────────────────")
        print("  HiSLM NX → AGX Messenger  (CLI mode)")
        print("  Type a message and press Enter to send.")
        print("  Ctrl-C to quit.")
        print("─────────────────────────────────────────\n")

        # Keepalive ping thread
        def _pinger():
            while not self._stop.is_set():
                time.sleep(15)
                if self._connected.is_set():
                    self.ping()

        threading.Thread(target=_pinger, daemon=True).start()

        try:
            while not self._stop.is_set():
                try:
                    line = input(f"[{self.name}] > ")
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                if line.lower() in ("/quit", "/exit", "/q"):
                    break
                if line.lower() == "/ping":
                    self.ping()
                    continue
                self.send(line)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.close()


# ─────────────────────────────────────────────
# GUI mode helper
# ─────────────────────────────────────────────

def open_browser_ui(base_url: str, client_id: str):
    """
    In GUI mode the server already hosts the full chat UI.
    We just open it in the local browser with client_id baked in.
    """
    url = f"{base_url}/?client_id={client_id}"
    log.info(f"[GUI] Opening browser: {url}")
    print(f"\n  Opening UI at: {url}")
    print("  If browser doesn't open, paste the URL manually.\n")
    webbrowser.open(url)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="HiSLM NX client — connects to AGX server"
    )
    p.add_argument("--agx-ip", required=True,
                   help="IP address of the AGX Orin (e.g. 172.16.6.21)")
    p.add_argument("--port", type=int, default=8000,
                   help="Server port (default: 8000)")
    p.add_argument("--name", default="nx-node",
                   help="Client identifier shown in chat (default: nx-node)")
    p.add_argument("--cli", action="store_true",
                   help="Run in terminal mode instead of opening browser UI")
    return p.parse_args()


def main():
    args = parse_args()
    base_url = f"http://{args.agx_ip}:{args.port}"
    ws_url   = f"ws://{args.agx_ip}:{args.port}/ws?client_id={args.name}"

    log.info("=" * 60)
    log.info("  HiSLM Node Messenger — NX Client")
    log.info(f"  AGX server  : {base_url}")
    log.info(f"  WebSocket   : {ws_url}")
    log.info(f"  Client name : {args.name}")
    log.info(f"  Mode        : {'CLI' if args.cli else 'GUI (browser)'}")
    log.info("=" * 60)

    # Health check first
    print(f"\n  Checking AGX server at {base_url}…")
    if not check_health(base_url):
        log.critical("[STARTUP] AGX server unreachable. Aborting.")
        print("\n✗ Cannot reach AGX server.")
        print(f"  Make sure server.py is running on {args.agx_ip}:{args.port}")
        print("  and both devices are on the same LAN.\n")
        sys.exit(1)
    print("  ✓ AGX server is online\n")

    if args.cli:
        # ── Terminal mode ─────────────────────
        client = WSCliClient(ws_url=ws_url, client_name=args.name)
        print(f"  Connecting WebSocket to {ws_url}…")
        if not client.connect(timeout=10):
            log.error("[STARTUP] WebSocket connection failed, falling back to REST poll")
            print("\n  WebSocket failed — falling back to REST polling mode.\n")
            _rest_cli_loop(base_url=base_url, name=args.name)
        else:
            client.run_cli()
    else:
        # ── GUI mode — just open the browser ──
        open_browser_ui(base_url, client_id=args.name)
        print("  Server is handling the UI. Keep this terminal open or close it.")
        print("  Ctrl-C to exit.\n")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass


def _rest_cli_loop(base_url: str, name: str):
    """
    REST-only fallback: poll /messages every 3s, send via POST /send.
    Used when WebSocket is unavailable.
    """
    log.info("[REST FALLBACK] entering REST CLI loop")
    print("\n─── REST Fallback Mode ────────────────────")
    print("  WebSocket unavailable. Using REST polling.")
    print("  Type a message and Enter to send. Ctrl-C to quit.\n")

    seen_ids: set = set()
    last_fetch = 0.0
    POLL_INTERVAL = 3.0

    def _poll():
        nonlocal last_fetch
        now = time.time()
        if now - last_fetch < POLL_INTERVAL:
            return
        last_fetch = now
        try:
            msgs = rest_get_messages(base_url, limit=50)
            for m in msgs:
                mid = m.get("id", "")
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    ts = m.get("timestamp", "")[:19].replace("T", " ")
                    sender = m.get("sender", "?")
                    role_tag = "[AGX]" if m.get("role") == "server" else "[NX] "
                    print(f"\r  {ts}  {role_tag}  {sender}: {m.get('text','')}")
        except Exception as exc:
            log.warning(f"[REST POLL] error: {exc}")

    try:
        while True:
            _poll()
            try:
                text = input(f"[{name}] > ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() in ("/quit", "/exit", "/q"):
                break
            try:
                rest_send(base_url, sender=name, text=text)
            except Exception as exc:
                print(f"  ✗ Send failed: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    log.info("[REST FALLBACK] loop exited")


if __name__ == "__main__":
    main()