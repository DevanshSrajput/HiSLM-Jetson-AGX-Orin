# HiSLM - Jetson AGX Orin

Hierarchical Small Language Model inference on NVIDIA Jetson Orin.
Runs Qwen2.5-3B (Q4_K_M) via llama.cpp with CUDA acceleration,
served over Tailscale VPN to an Orin NX client.

## Hardware

| Device | Role | Specs |
|--------|------|-------|
| Jetson AGX Orin | Server (LLM inference) | 30 GB VRAM, 8x Cortex-A78AE |
| Jetson Orin NX | Client (wireless UI) | 8 GB RAM |

## Software Stack

- **Model**: [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) (GGUF Q4_K_M, 1.79 GiB)
- **Runtime**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (CUDA, sm87)
- **Server**: FastAPI + WebSocket + REST (server2.py)
- **Client**: CLI + GUI Web UI (client2.py)
- **Networking**: Tailscale VPN

## Performance

| Test | GPU (-ngl 99) | CPU only | Speedup |
|------|:-------------:|:--------:|:-------:|
| Prompt processing | 948 tok/s | 422 tok/s | 2.2x |
| Text generation | 24.9 tok/s | 4.17 tok/s | 6.0x |

All 36 layers offloaded to GPU. Temperatures below 42C under load.

## Quick Start

### Server (AGX Orin)

```bash
python server2.py
```

Auto-detects Tailscale IP and serves on `0.0.0.0:8000`.

### Client (Orin NX)

```bash
python client2.py --agx-ip <TAILSCALE_IP>
```

Opens a browser GUI. Add `--cli` for terminal mode.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `llama.cpp/models/Qwen2.5-3B-Q4_K_M.gguf` | GGUF model file |
| `LLAMA_RUNNER` | `llama.cpp/build/bin/llama-cli` | llama.cpp binary |
| `LLAMA_N_PREDICT` | `256` | Max tokens per response |
| `LLAMA_THREADS` | `8` | CPU threads for LLM |
| `LLAMA_CTX_SIZE` | `8192` | Context window |
| `LLAMA_GPU_LAYERS` | `99` | GPU layers (-ngl) |
| `LLAMA_TIMEOUT_SECONDS` | `300` | Inference timeout |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| WebSocket | `/ws?client_id=<name>` | Real-time chat |
| POST | `/send` | Send message (REST fallback) |
| GET | `/messages` | Message history |
| GET | `/health` | Server status |
| GET | `/` | Web UI |
| GET | `/nx` | NX client UI |

## Files

```
HiSLM/
  server2.py          AGX inference server (Tailscale + WebSocket)
  client2.py          NX wireless client (GUI or CLI)
  static/
    index.html        Server web UI
    nx_index.html     NX client web UI
  llama.cpp/          llama.cpp submodule (GPU build)
  Qwen2.5-3B-benchmark.md  Benchmark results
```
