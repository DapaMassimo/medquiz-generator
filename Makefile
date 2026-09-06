.PHONY: install lint format type test test-unit test-int test-eval check run cli clean hooks licenses

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

# Attribution bundle for delivery (Docker image, installer, source drop).
# Runtime dependencies only: dev tools are never shipped, so they carry no
# attribution duty. Regenerate on the platform you deliver from, since the
# resolved dependency set differs between Linux and Windows.
licenses:
	@uv export --no-dev --no-hashes --no-emit-project --format requirements-txt \
	  | sed -n 's/^\([A-Za-z0-9._-]*\)==.*/\1/p' | tr '\n' ' ' > .licenses-pkgs.tmp
	@uv run pip-licenses --from=mixed --with-authors --with-urls --with-license-file \
	  --no-license-path --format=plain-vertical \
	  --packages $$(cat .licenses-pkgs.tmp) --output-file=THIRD_PARTY_LICENSES.txt
	@rm -f .licenses-pkgs.tmp
	@echo "Wrote THIRD_PARTY_LICENSES.txt"
