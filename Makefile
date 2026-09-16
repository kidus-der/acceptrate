.PHONY: test lint fmt smoke reproduce

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

reproduce:
	@echo "make reproduce is a P7 deliverable; nothing to regenerate yet" >&2
	@exit 1
