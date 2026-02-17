# Docker-First LoRA Fine-Tuning Pipeline for ASUS TURBO Radeon™ AI PRO R9700

A Docker-first LoRA fine-tuning workflow for AMD ROCm systems, designed to showcase the ASUS TURBO Radeon AI PRO R9700 from a clean Ubuntu 24.04.4 server.

This project is intentionally phase-based so operators can run, debug, and hand over the workflow with minimal ambiguity.

## What this repository guarantees

1. A strict phase sequence from setup to local chat.
2. Docker-based execution for preparation, training, and comparison.
3. Early failure for environment issues via preflight checks.
4. Predictable output locations for data, model artefacts, and reports.
5. Consistent container timezone across Docker phases (default `Asia/Kuala_Lumpur`).

## Why Docker is mandatory for fine-tuning phases

ROCm pipelines often fail due to host drift (package versions, Python environments, and ROCm/PyTorch mismatch). Docker is used here to keep the runtime consistent for the compute-critical phases:

1. `scripts/phase1_prepare_data.sh`
2. `scripts/phase2_training.sh`
3. `scripts/phase3_comparison_report.sh`

Phase 5 is local (`llama.cpp` chat). Phase 4 uses Docker for merge and conversion, then uses local `llama.cpp` quantisation binary if available.
For Phase 0 to Phase 4, scripts also pass timezone settings into `docker run` so logs and timestamps stay consistent.

## Official AMD ROCm Docker reference used by this project

This repository follows AMD ROCm official PyTorch installation guidance:

1. PyTorch on ROCm installation:
   - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html
2. Docker with PyTorch pre-installed:
   - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-docker-with-pytorch-pre-installed

The project default base image in `scripts/phase0_prelim.sh` is:
- `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1`

Why this pinned tag is used instead of `latest`:
1. Better reproducibility for demos and troubleshooting.
2. Clear alignment with a validated ROCm/PyTorch inventory.
3. Reduced drift risk when upstream tags move.

You can still override it when needed:

```bash
export BASE_IMAGE="rocm/pytorch:latest"
./scripts/start_finetuning_process.sh prelim
```

or pin another AMD-supported tag:

```bash
export BASE_IMAGE="rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1"
./scripts/start_finetuning_process.sh prelim
```

## Why this base model is used

Default base model:
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

Reasoning:
1. It is a strong instruction-tuned baseline.
2. It supports LoRA adaptation well in the HF + PEFT ecosystem.
3. It is practical for adapter training, merge, and GGUF export.
4. It gives high baseline capability, while your dataset shifts behaviour towards your target use case.

## LoRA in this project (practical explanation)

LoRA does not retrain full model weights. It trains small low-rank adapters attached to selected transformer layers.

What that means operationally:
1. Training is cheaper and faster than full fine-tuning.
2. The base model remains unchanged.
3. The main training output is an adapter directory (`models/lora_*`).
4. You can later merge adapter + base into a standalone model (Phase 4).

## Repository layout

```text
.
├── README.md
├── requirements.txt
└── scripts/
    ├── cli_theme.sh
    ├── verify_environment.sh
    ├── start_finetuning_process.sh
    ├── phase0_prelim.sh
    ├── phase1_prepare_data.sh
    ├── phase2_training.sh
    ├── phase3_comparison_report.sh
    ├── phase4_export_gguf.sh
    └── phase5_chat_llamacpp.sh
```

Generated at runtime:
- `data/`
- `models/`
- `results/`
- `logs/`

## Core entry points

Dispatcher:
- `scripts/start_finetuning_process.sh`

Preflight checker:
- `scripts/verify_environment.sh`

Supported dispatcher commands:

```bash
./scripts/start_finetuning_process.sh prelim
./scripts/start_finetuning_process.sh prepare
./scripts/start_finetuning_process.sh train
./scripts/start_finetuning_process.sh compare
./scripts/start_finetuning_process.sh gguf
./scripts/start_finetuning_process.sh chat
```

The dispatcher forwards additional arguments to phase scripts. Example:

```bash
./scripts/start_finetuning_process.sh train --full
./scripts/start_finetuning_process.sh train --background --full
./scripts/start_finetuning_process.sh chat -- -n 256
```

## Prerequisites

Mandatory:
1. Ubuntu 24.04.x recommended.
2. ROCm-capable AMD GPU with `/dev/kfd` and `/dev/dri` available.
3. Docker installed and usable by your current user.
4. Hugging Face access approval for `meta-llama/Meta-Llama-3.1-8B-Instruct`.
5. Hugging Face token.

Required for later phases:
1. `llama.cpp` checkout for Phase 4 conversion and Phase 5 chat.
2. `llama.cpp` build with `llama-cli` (Phase 5) and optionally `llama-quantize` (Phase 4 quantisation step).

Optional local Python dependencies (only if you run custom host-side scripts):

```bash
python3 -m pip install -r requirements.txt
```

## Complete from-zero setup (Ubuntu 24.04.4)

Follow these commands in order on a brand new server.

### 1) Install base packages

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release git build-essential cmake python3-pip
```

### 2) Install Docker Engine (official Docker repository)

```bash
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Add your user to Docker group and apply it:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker --version
docker info
```

Optional validation against AMD ROCm docs before running prelim:

```bash
docker pull rocm/pytorch:latest
docker pull rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

Then run:

```bash
./scripts/start_finetuning_process.sh prelim
```

`prelim` will pull/build using `BASE_IMAGE` automatically.

### 3) Clone this project and fix script permissions

```bash
cd ~
git clone https://github.com/pendakwahteknologi/finetune-rocm.git
cd finetune-rocm
chmod +x scripts/*.sh
```

Why `chmod` is included:
1. Some clone/archive workflows can drop executable bits.
2. This prevents `Permission denied` when running `./scripts/*.sh`.

### 4) Set Hugging Face authentication

```bash
export HF_TOKEN="<your_hf_token>"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Optional CLI login:

```bash
huggingface-cli login
```

### 5) Set container timezone (recommended)

Default is already `Asia/Kuala_Lumpur`. Set it explicitly so all shells and runs stay consistent:

```bash
export CONTAINER_TZ="Asia/Kuala_Lumpur"
echo 'export CONTAINER_TZ="Asia/Kuala_Lumpur"' >> ~/.bashrc
source ~/.bashrc
```

If you need a different timezone, change `CONTAINER_TZ` before running any phase.

### 6) Install and build llama.cpp (required for Phase 4 and Phase 5)

Recommended location:
1. Keep it in your home folder: `~/llama.cpp`.
2. Reuse the same build across multiple projects.

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Set environment variable:

```bash
echo 'export LLAMA_CPP_DIR="$HOME/llama.cpp"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
test -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" && echo converter_ok
test -x "$LLAMA_CPP_DIR/build/bin/llama-cli" && echo cli_ok
```

### 7) Run preflight and full workflow

```bash
cd ~/finetune-rocm
./scripts/verify_environment.sh

./scripts/start_finetuning_process.sh prelim
./scripts/start_finetuning_process.sh prepare
./scripts/start_finetuning_process.sh train --full
./scripts/start_finetuning_process.sh compare
./scripts/start_finetuning_process.sh gguf
./scripts/start_finetuning_process.sh chat
```

Important:
1. `prelim` builds/pulls the Docker image (`finetune-rocm:ready`) automatically.
2. You do not need to build the training image manually.
3. Run commands from the repository root (`~/finetune-rocm`).

## Security and authentication

Never hardcode secrets in scripts.

Set token variables in your shell:

```bash
export HF_TOKEN="<your_hf_token>"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Why both:
1. Different tools read different variable names.
2. Setting both removes ambiguity.

Optional CLI login (for manual Hugging Face operations):

```bash
huggingface-cli login
```

Run all project commands from the repository root directory.

## Phase 0 preflight: what it checks

`verify_environment.sh` is now part of prelim and can also be run directly.

Command:

```bash
./scripts/verify_environment.sh
```

Checks performed:
1. OS profile (`/etc/os-release`).
2. Docker binary and daemon reachability.
3. ROCm device nodes (`/dev/kfd`, `/dev/dri`).
4. User groups (`video`, `render`).
5. HF token presence.
6. Network reachability to `huggingface.co`.
7. Free disk space (`MIN_FREE_GB`, default `60`).
8. `llama.cpp` converter and chat binary presence.

Failure policy:
1. Critical issues stop execution.
2. Warnings are non-blocking.

## End-to-end quick start

```bash
export HF_TOKEN="<your_hf_token>"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export CONTAINER_TZ="Asia/Kuala_Lumpur"

./scripts/start_finetuning_process.sh prelim
./scripts/start_finetuning_process.sh prepare
./scripts/start_finetuning_process.sh train --full
./scripts/start_finetuning_process.sh compare
./scripts/start_finetuning_process.sh gguf
./scripts/start_finetuning_process.sh chat
```

## Phase-by-phase guide

### Phase 0: Prelim (`scripts/phase0_prelim.sh`)

Command:

```bash
./scripts/start_finetuning_process.sh prelim
```

Actions:
1. Runs environment verification (`verify_environment.sh`).
2. Pulls/builds training image if needed.
3. Caches base model snapshot in HF cache.

Outputs:
- Docker image: `finetune-rocm:ready` (or `IMAGE_NAME`)
- HF cache directory (default `~/.cache/huggingface`)

### Phase 1: Data preparation (`scripts/phase1_prepare_data.sh`)

Command:

```bash
./scripts/start_finetuning_process.sh prepare
```

Actions:
1. Loads Dolly and OpenOrca sample.
2. Cleans rows and removes chain-of-thought trigger prompts.
3. Splits eval prompts.
4. Converts training rows into ChatML messages.

Outputs:
- `data/dolly15k_plus_openorca<N>.jsonl`
- `data/train_clean.jsonl`
- `data/eval_prompts.jsonl`
- `data/train_chatml.jsonl`
- `logs/clean_report.txt`

### Phase 2: Training (`scripts/phase2_training.sh`)

Default command:

```bash
./scripts/start_finetuning_process.sh train
```

Modes:

```bash
./scripts/phase2_training.sh
./scripts/phase2_training.sh --full
./scripts/phase2_training.sh --background --full
./scripts/phase2_training.sh --status
./scripts/phase2_training.sh --stop
```

Outputs:
- quick run adapter: `models/lora_simple_rocm_out/`
- full run adapter: `models/lora_full_20k_rocm_out/`
- background log: `results/training.log`

### Phase 3: Comparison + HTML report (`scripts/phase3_comparison_report.sh`)

Command:

```bash
./scripts/start_finetuning_process.sh compare
```

Actions:
1. Runs prompt set through base and fine-tuned models.
2. Stores structured JSON results.
3. Generates human-readable HTML report.

Outputs:
- `results/comparison_results.json`
- `results/comparison_report.html`

### Phase 4: Merge + GGUF export (`scripts/phase4_export_gguf.sh`)

Command:

```bash
./scripts/start_finetuning_process.sh gguf
```

Actions:
1. Runs merge and GGUF conversion inside Docker (`finetune-rocm:ready`), so host `torch` is not required.
2. Loads base model and LoRA adapter, then merges adapter into base weights.
3. Converts merged model to GGUF via `llama.cpp/convert_hf_to_gguf.py`.
4. Optionally quantises GGUF using local `llama-quantize` binary if present.

Outputs:
- `models/merged_llama31_8b_lora/`
- `models/gguf/model-f16.gguf`
- `models/gguf/model-<quant>.gguf`

### Phase 5: Local chat via llama.cpp (`scripts/phase5_chat_llamacpp.sh`)

Command:

```bash
./scripts/start_finetuning_process.sh chat
```

Actions:
1. Uses `model-Q8_0.gguf` if present, else `model-f16.gguf`.
2. Starts `llama-cli` (or legacy `main`) in conversation mode.
3. Lets you chat with your fine-tuned model locally.

## Output interpretation

### `models/lora_*`
1. LoRA adapter weights and tokenizer files.
2. Not a full standalone model.
3. Used for comparison and Phase 4 merge.

### `results/comparison_results.json`
1. Machine-readable A/B output data.
2. Includes token/time metrics and response pairs.
3. Useful for regression tracking.

### `results/comparison_report.html`
1. Human-readable summary of response differences.
2. Useful for demos and stakeholder review.

### `models/gguf/*.gguf`
1. Standalone inference artefacts for llama.cpp.
2. Quantised files are smaller/faster but may reduce output quality.

## Environment variable reference

### Global/auth

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `HF_TOKEN` | unset | phases 0,2,3,4 + verify | HF model access |
| `HUGGINGFACE_HUB_TOKEN` | unset | phases 0,2,3,4 + verify | Alternate HF token variable |
| `HF_CACHE_DIR` | `~/.cache/huggingface` (phase 4 defaults to `<repo>/.cache/huggingface`) | phases 0,1,2,3,4 | Shared HF cache |

### Preflight and prelim

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `MIN_FREE_GB` | `60` | `verify_environment.sh` | Minimum recommended free disk |
| `BASE_IMAGE` | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` | `phase0_prelim.sh` | Docker base image (AMD ROCm PyTorch tag) |
| `IMAGE_NAME` | `finetune-rocm:ready` | phases 0,1,2,3,4 | Runtime image tag |
| `CONTAINER_TZ` | `Asia/Kuala_Lumpur` | phases 0,1,2,3,4 | Container timezone for logs/timestamps |
| `BASE_MODEL` | Llama 3.1 8B instruct | phases 0,3,4 | Base HF model ID |
| `FORCE_REBUILD` | `0` | `phase0_prelim.sh` | Force Docker image rebuild |
| `LLAMA_CPP_DIR` | `<repo>/llama.cpp` | verify, phases 4,5 | llama.cpp root path |

### Data prep

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `OPENORCA_N` | `5000` | `phase1_prepare_data.sh` | OpenOrca sample size |
| `EVAL_N` | `200` | `phase1_prepare_data.sh` | Eval prompt count |
| `SEED` | `42` | phase1 + phase2(in-container) | Random seed |
| `SYSTEM_PROMPT` | `You are a helpful assistant.` | `phase1_prepare_data.sh` | ChatML system message |

### Training

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `MAX_TRAIN_ROWS` | `2000` | `phase2_training.sh` | Quick-mode row cap |
| `MAX_TRAIN_ROWS_FULL` | `20011` | `phase2_training.sh` | Full-mode row cap |
| `OUT_DIR` | `/workspace/models/lora_simple_rocm_out` | `phase2_training.sh` | Quick-mode output |
| `OUT_DIR_FULL` | `/workspace/models/lora_full_20k_rocm_out` | `phase2_training.sh` | Full-mode output |
| `BATCH_SIZE` | `1` | `phase2_training.sh` | Batch size |
| `GRAD_ACCUM` | `4` | `phase2_training.sh` | Gradient accumulation steps |
| `USE_FP16` | `1` | `phase2_training.sh` | Enable fp16 |
| `USE_BF16` | `0` | `phase2_training.sh` | Enable bf16 |
| `DATALOADER_WORKERS` | `0` | `phase2_training.sh` | Data loader workers |
| `TRAIN_LOG` | `results/training.log` | `phase2_training.sh` | Background log path |
| `CONTAINER_NAME` | `rocm-training` | `phase2_training.sh` | Background container name |

Advanced training variables passed through to in-container Python include `EPOCHS`, `SEQ_LEN`, `LR`, `LORA_R`, and `LORA_ALPHA`.
Phase 2 uses `meta-llama/Meta-Llama-3.1-8B-Instruct` by default inside the embedded trainer. To change model family for Phase 2, edit `scripts/phase2_training.sh`.

### GGUF export and chat

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `LORA_ADAPTER` | `models/lora_simple_rocm_out` | phases 3,4 | Adapter path |
| `MERGED_DIR` | `models/merged_llama31_8b_lora` | phase4 | Merged model output |
| `GGUF_OUT_DIR` | `models/gguf` | phase4 | GGUF output directory |
| `GGUF_OUTTYPE` | `f16` | phase4 | GGUF conversion type |
| `QUANTIZE_TYPE` | `Q8_0` | phase4 | Quantisation target |
| `SKIP_QUANTIZE` | `0` | phase4 | Skip quantisation step |
| `MERGE_DTYPE` | `float16` | phase4 | Merge dtype (`float16`, `bfloat16`, `float32`) |
| `MODEL_GGUF` | `<repo>/models/gguf/model-Q8_0.gguf` | phase5 | Preferred chat model file |
| `MODEL_F16` | `<repo>/models/gguf/model-f16.gguf` | phase5 | Fallback chat model |
| `CTX_SIZE` | `8192` | phase5 | Context window |
| `N_GPU_LAYERS` | `99` | phase5 | Offloaded layers |
| `TEMP` | `0.7` | phase5 | Sampling temperature |
| `TOP_P` | `0.9` | phase5 | Nucleus sampling |
| `THREADS` | auto-detected | phase5 | CPU thread count |

## Troubleshooting

### `Docker is required but not found in PATH`
1. Install Docker.
2. Confirm with `docker --version`.

### `Docker daemon is not reachable for current user`
1. Start Docker service.
2. Ensure your user has permission to access Docker.

### `HF token missing` or model access errors (401/403)
1. Export `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`.
2. Confirm your account has approval for Meta Llama model.

### `ROCm device missing: /dev/kfd` or `/dev/dri`
1. Validate ROCm install on host.
2. Confirm device nodes exist and permissions are correct.

### `Missing converter: .../convert_hf_to_gguf.py`
1. Set `LLAMA_CPP_DIR` correctly.
2. Ensure llama.cpp repository is present and built.

### `ValueError: Failed to detect model architecture` during GGUF conversion
1. Re-run Phase 4 with clean merged output:

```bash
CLEAN_MERGED_DIR=1 ./scripts/start_finetuning_process.sh gguf
```

2. Ensure `llama.cpp` is up to date and rebuilt:

```bash
cd "$LLAMA_CPP_DIR"
git pull
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

### `mkdir: cannot create directory 'models/gguf': Permission denied`
1. This is usually ownership drift from Docker-created files.
2. From repo root, fix ownership:

```bash
sudo chown -R "$USER":"$USER" .
```

3. Re-run:

```bash
./scripts/start_finetuning_process.sh gguf
```

### Phase 5 cannot find GGUF
1. Run Phase 4 first.
2. Or set `MODEL_GGUF` to an existing GGUF path.

### Container time is wrong
1. Check active setting: `echo "$CONTAINER_TZ"`.
2. Set it explicitly (example): `export CONTAINER_TZ="Asia/Kuala_Lumpur"`.
3. Re-run the phase command so a new container starts with the correct timezone.

## Fresh-server validation sequence

Run this on a new Ubuntu machine to validate readiness quickly:

```bash
./scripts/verify_environment.sh
./scripts/start_finetuning_process.sh prelim
./scripts/start_finetuning_process.sh prepare
./scripts/start_finetuning_process.sh train --full
./scripts/start_finetuning_process.sh compare
./scripts/start_finetuning_process.sh gguf
./scripts/start_finetuning_process.sh chat
```
