.PHONY: sync test test-slow lint format typecheck bench notebooks llms docs build

sync:
	uv sync --locked --all-extras --group dev --group docs

test:
	uv run pytest scspill/tests -n auto

test-slow:
	uv run pytest scspill/tests -m slow

lint:
	uv run ruff check scspill tools benchmarks
	uv run ruff format --check scspill tools benchmarks

format:
	uv run ruff format scspill tools benchmarks
	uv run ruff check --fix scspill tools benchmarks

typecheck:
	uv run mypy scspill

bench:
	uv run python benchmarks/run_benchmarks.py --all --report

notebooks:
	uv run python tools/build_colab_notebooks.py

llms:
	uv run python tools/build_llms_txt.py

docs: notebooks
	uv run quartodoc build --config docs/_quarto.yml
	uv run python -m ipykernel install --user --name scspill
	uv run quarto render docs
	uv run python tools/build_llms_txt.py

build:
	uv build
