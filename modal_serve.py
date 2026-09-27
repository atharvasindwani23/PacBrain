"""vLLM OpenAI-compatible servers for the base and tuned models.
Deploy:  modal deploy modal_serve.py
Then hit <url>/v1/completions with model name "pacman".
"""

import subprocess
from pathlib import Path

import modal

BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
PORT = 8000

app = modal.App("pacbrain-serve")
vol = modal.Volume.from_name("pacbrain-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.7.2", "transformers==4.48.2", "hf-transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "0"})
)


def _launch(model_path: str, name: str):
    cmd = ["vllm", "serve", model_path, "--served-model-name", name,
           "--port", str(PORT), "--max-model-len", "1024",
           "--gpu-memory-utilization", "0.90", "--enforce-eager"]
    subprocess.Popen(cmd)


@app.function(image=image, gpu="A10G", volumes={"/vol": vol},
              scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=600)
def serve_base():
    _launch(BASE_MODEL, "pacman")


@app.function(image=image, gpu="A10G", volumes={"/vol": vol},
              scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=600)
def serve_tuned():
    # Training commits the volume separately; refresh before opening files.
    vol.reload()
    checkpoint = Path("/vol/tuned")
    if not (checkpoint / "config.json").is_file() or not (
            list(checkpoint.glob("*.safetensors")) or list(checkpoint.glob("pytorch_model*.bin"))):
        raise RuntimeError("No merged checkpoint at /vol/tuned; finish training and commit the volume first")
    _launch("/vol/tuned", "pacman")


@app.function(image=image, gpu="A10G", volumes={"/vol": vol},
              scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=600)
def serve_rl():
    _launch("/vol/rl", "pacman")


@app.function(image=image, gpu="A10G", volumes={"/vol": vol},
              scaledown_window=600, timeout=3600)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=600)
def serve_rl2():
    _launch("/vol/rl2", "pacman")
