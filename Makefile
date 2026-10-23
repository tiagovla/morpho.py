.PHONY: build docs serve

build:
	uv build

docs:
	uv run mkdocs build

serve:
	uv run mkdocs serve
