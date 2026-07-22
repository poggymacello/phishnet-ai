.PHONY: install lint test run clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest -q

run:
	.venv/bin/python -m phishnet train

clean:
	rm -rf .venv **/__pycache__ .pytest_cache .ruff_cache
