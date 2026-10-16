# Start a vLLM server in Docker for chat expansion (engine: vllm in configs/generators.yaml).
# Uses the local Hugging Face cache, offline, so the pinned revision must be downloaded first.
# Stop it with: docker rm -f vllm
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/serve_vllm.ps1
param(
    [string]$Model = "Qwen/Qwen3-8B-AWQ",
    [string]$Revision = "4da05a8edb55c6046cce958586c33b61da07bb79",
    [string]$Image = "vllm/vllm-openai:v0.30.0",
    # Longest prompt (product text cut to 384 tokens + instructions) plus output fits in 2,048.
    [int]$MaxModelLen = 2048,
    [int]$MaxNumSeqs = 128
)
$ErrorActionPreference = "Stop"
docker ps -aq --filter "name=^vllm$" | ForEach-Object { docker rm -f $_ | Out-Null }
docker run -d --name vllm --gpus all --ipc=host -p 8000:8000 `
    -v "$env:USERPROFILE\.cache\huggingface:/root/.cache/huggingface" `
    -e HF_HUB_OFFLINE=1 `
    $Image `
    --model $Model --revision $Revision `
    --max-model-len $MaxModelLen --max-num-seqs $MaxNumSeqs `
    --gpu-memory-utilization 0.90 --seed 0
