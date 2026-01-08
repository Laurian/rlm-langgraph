.PHONY: install lint format test clean

install:
	uv sync

install-dev:
	uv sync --extra dev

lint:
	uv run ruff check src/ examples/
	uv run ruff format --check src/ examples/

format:
	uv run ruff check --fix src/ examples/
	uv run ruff format src/ examples/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=rlm_langgraph --cov-report=html

clean:
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

example-basic:
	uv run python examples/basic_usage.py

example-needle:
	uv run python examples/needle_in_haystack.py

example-doc:
	uv run python examples/long_document.py
