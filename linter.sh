#!/bin/bash
# linter.sh - uruchamia narzędzia do formatowania i sprawdzania kodu Python

# Wyjście na czerwono w przypadku błędu
set -e

echo "=== Running isort (sort imports) ==="
isort .

echo "=== Running black (autoformat) ==="
black .

echo "=== Running flake8 (linting) ==="
flake8 .

echo "✅ All checks passed!
