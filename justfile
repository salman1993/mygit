# Setup the project with `uv init --build-backend uv <YOUR_PROJECT>`


format:
    @echo "→ Formatting code with ruff"
    uv run ruff format src/

lint:
    @echo "→ Linting code issues with ruff (with auto-fix)"
    -uv run ruff check --fix src/

typecheck:
    @echo "→ Type checking with ty"
    uv run ty check

check: format lint typecheck
    @echo "→ All checks completed"
