.PHONY: test lint fmt smoke reproduce tokens tokens-check calibrate lossless sweep-one adaptive tui tui-test chat-demo

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

TRACES ?= traces/sweep
FIGURES ?= docs/figures

tui:  ## build the Go TUI binary into tui/bin
	cd tui && go build -o bin/acceptrate-tui ./cmd/acceptrate-tui

tui-test:
	cd tui && go vet ./... && go test ./...

chat-demo: tui  ## the TUI against its in-process mock, no model needed
	tui/bin/acceptrate-tui --mock

reproduce:  ## regenerate every published figure and table from raw traces
	uv run acceptrate analysis p3-gate $(TRACES)
	uv run acceptrate analysis fit $(TRACES) --out $(FIGURES)
	uv run acceptrate analysis figures $(TRACES) --out $(FIGURES)
