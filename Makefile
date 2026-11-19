.PHONY: install install-mamba install-vllm serve test test-integration lint fmt typecheck check

install:
	uv sync --extra dev --extra skillopt --extra lora

# Prebuilt Mamba2 kernels for the Nemotron hybrid. causal_conv1d and mamba_ssm
# have no PyPI wheel for this torch, and building either from source needs an
# nvcc the nodes do not default to, so these are the release wheels for cu12,
# torch 2.9, the C++11 ABI and whichever interpreter .venv was made with.
# --no-deps: mamba-ssm asks for bare torch, tilelang and quack-kernels;
# resolving those could move the cu128 torch, and the last two are optional
# imports the Mamba2 path never reaches. `uv sync` prunes both packages (they
# are not in the lockfile) while `uv run` leaves them alone, so this has to be
# run again after `make install`.
# --python, because `uv pip install` honours an active VIRTUAL_ENV while
# `uv run` ignores it and uses .venv: without it the wheels can land in an
# environment the job never looks at, which is invisible until the mixer
# falls back mid-run.
VENV_PY = .venv/bin/python
CP = $(shell $(VENV_PY) -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')

install-mamba:
	uv pip install --python $(VENV_PY) --no-deps \
	  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-$(CP)-$(CP)-linux_x86_64.whl' \
	  'https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-$(CP)-$(CP)-linux_x86_64.whl'

# vLLM pins torch==2.13.0, which would fight this project's cu128 pin, so it
# gets its own environment and is reached over HTTP. cu129, not the default
# cu130: the cluster driver is 12.8, and cu129 runs on it by minor-version
# compatibility.
# --torch-backend only redirects the torch family, so vLLM's own wheel has to
# be asked for by name: PyPI's vllm==0.27.1 is the CUDA 13 build, and its
# compiled extension then fails to import with libcudart.so.13 not found.
# transformers is pinned below 5.15, which moved Gemma 4's head_dim into a
# per-layer config that this vllm still reads globally and dies on. 5.14.1 is
# the schema the 0.27.1 wheel was built against; Qwen3.5 serves unchanged.
install-vllm:
	uv venv --python 3.12 .venv-vllm
	uv pip install --python .venv-vllm --torch-backend=cu129 \
	  --extra-index-url https://wheels.vllm.ai/0.27.1/cu129/ \
	  'vllm==0.27.1+cu129' 'transformers==5.14.1'

# MODEL=/path/to/checkpoint make serve
serve:
	.venv-vllm/bin/vllm serve $(MODEL) --port 8000 --max-model-len 20480 \
	  --gpu-memory-utilization 0.90

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests baselines/llmskill baselines/skillopt baselines/lora baselines/ultrabreak

fmt:
	uv run ruff format src tests baselines/llmskill baselines/skillopt baselines/lora baselines/ultrabreak
	uv run ruff check --fix src tests baselines/llmskill baselines/skillopt baselines/lora baselines/ultrabreak

typecheck:
	uv run mypy

check: lint typecheck test
