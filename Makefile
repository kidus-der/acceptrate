.PHONY: test lint fmt smoke reproduce tokens tokens-check calibrate lossless sweep-one adaptive

test:
	uv run pytest -m "not model and not perf"

test-model:
	uv run pytest -m "model"

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

smoke:
	uv run acceptrate bench --smoke

calibrate:
	uv run acceptrate verify calibrate

lossless:
	uv run acceptrate verify lossless --prompts 20 --k 4 --max-tokens 128

sweep-one:  ## DRAFT=mlx-community/... make sweep-one
	uv run acceptrate bench sweep --draft $(DRAFT) --ks 0-8 --prompts-per-tag 6 --max-tokens 200

adaptive:
	uv run acceptrate bench adaptive

tokens:
	uv run python -m acceptrate.design --write

tokens-check:
	uv run python -m acceptrate.design --check

reproduce:
	@echo "make reproduce is a P7 deliverable; nothing to regenerate yet" >&2
	@exit 1
