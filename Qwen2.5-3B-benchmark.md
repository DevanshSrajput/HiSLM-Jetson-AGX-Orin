# Qwen2.5-3B Q4_K_M Benchmark

## System

| Component | Spec |
|-----------|------|
| SoC | NVIDIA Jetson Orin (nvgpu) |
| GPU | Orin, compute capability 8.7, 30 GB VRAM |
| CPU | 8x Cortex-A78AE @ max 2.19 GHz |
| RAM | 29 GB |
| CUDA | 12.6 |
| llama.cpp | b9571-e3471b3e7 |
| Quantization | Q4_K_M |
| Model Size | 1.79 GiB |
| Parameters | 3.09 B |

---

## Temperatures

| Sensor | Pre | Peak During | Post |
|--------|-----|-------------|------|
| CPU | 41.6C | 41.9C | 41.3C |
| GPU | 37.1C | 39.5C | 37.5C |
| SOC | 38.6C | 39.1C | 38.6C |
| TJ (Junction) | 41.7C | 42.0C | 41.9C |

Temperatures stayed well within safe limits throughout.

---

## CPU Frequencies

All 8 cores ran at ~1497 MHz during benchmark (max 2188 MHz).

---

## Benchmark: GPU Offloaded (`-ngl 99`)

| Test | Speed |
|------|-------|
| Prompt Processing (512 tokens) | **948.09 tok/s** |
| Text Generation (128 tokens) | **24.90 tok/s** |

Backend: CUDA (all 36 layers offloaded to GPU)

---

## Benchmark: CPU Only (`-ngl 0`)

| Test | Speed |
|------|-------|
| Prompt Processing (512 tokens) | 422.36 tok/s |
| Text Generation (128 tokens) | 4.17 tok/s |

Backend: CUDA (0 layers offloaded)

---

## Speedup with GPU

| Metric | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| Prompt Processing | 422 tok/s | 948 tok/s | **2.2x** |
| Text Generation | 4.17 tok/s | 24.90 tok/s | **6.0x** |

---

## Memory Usage

| State | Used | Available |
|-------|------|-----------|
| Pre-benchmark | 8.9 GB | 20 GB |
| Post-benchmark | 8.8 GB | 20 GB |
