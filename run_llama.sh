#!/bin/bash
export LD_LIBRARY_PATH=/home/nvidia/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH
exec /home/nvidia/HiSLM/llama.cpp/build/bin/llama-cli "$@"
