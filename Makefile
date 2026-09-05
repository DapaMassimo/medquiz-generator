.PHONY: install lint format type test test-unit test-int test-eval check run cli clean hooks

install:
	uv sync --all-groups

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports

format:
	uv run ruff format .
	uv run ruff check --fix .

type:
	uv run mypy

test:
	uv run pytest --cov=medquiz_generator --cov-report=term-missing --cov-fail-under=80

test-unit:
	uv run pytest tests/unit

test-int:
	uv run pytest tests/integration

test-eval:
	uv run pytest -m eval -o addopts=""

check: lint type test

run:
	uv run streamlit run src/medquiz_generator/entrypoints/streamlit_app.py

cli:
	uv run medquiz

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

hooks:
	uv run pre-commit install
