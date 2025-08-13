# MINTO Documentation Makefile

.PHONY: help docs-build docs-en docs-ja docs-serve docs-clean

help: ## Show this help message
	@echo "MINTO Documentation Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

docs-build: ## Build all documentation (simple version)
	@./scripts/simple-build.sh

docs-en: ## Build English documentation only
	@echo "Building English documentation..."
	@cd docs/en && uv run jupyter-book build .
	@echo "✅ Built: docs/en/_build/html/index.html"

docs-ja: ## Build Japanese documentation only
	@echo "Building Japanese documentation..."
	@cd docs/ja && uv run jupyter-book build .
	@echo "✅ Built: docs/ja/_build/html/index.html"

docs-serve: ## Build all and serve at localhost:8000
	@./scripts/simple-build.sh
	@echo "Starting server at http://localhost:8000"
	@cd _site && python -m http.server 8000

docs-clean: ## Clean all build artifacts
	@echo "Cleaning build artifacts..."
	@rm -rf docs/en/_build docs/ja/_build _site
	@echo "✅ Cleaned"