.PHONY: help install dev test lint format typecheck clean run-example

help:
	@echo "Tacit Video Annotator — common targets"
	@echo ""
	@echo "  make install       Install package in editable mode"
	@echo "  make dev           Install with dev dependencies"
	@echo "  make test          Run pytest"
	@echo "  make lint          Run ruff lint"
	@echo "  make format        Run ruff format"
	@echo "  make typecheck     Run mypy"
	@echo "  make clean         Remove build artifacts"
	@echo "  make run-example   Run pipeline on the example video (requires ANTHROPIC_API_KEY)"
	@echo ""

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-example:
	tacit-annotate run examples/sample_video.mp4 --branch pcr --output-dir ./out
