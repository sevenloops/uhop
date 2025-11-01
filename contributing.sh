#!/usr/bin/env bash
set -euo pipefail

# Contributing helper for UHOP
#
# Usage:
#   ./contributing.sh [command]
#
# Commands:
#   help             Show this help
#   setup            Create venv and install dependencies (dev)
#   test             Run Python unit tests (fast)
#   test:all         Run full Python tests (including benchmarks if enabled)
#   lint             Run basic linters/checks (ruff/flake8 if available, markdownlint-cli2 if installed)
#   fmt              Run formatters (black/isort/prettier if available)
#   hooks            Install pre-commit hooks (if pre-commit is installed)
#   frontend:dev     Install and run the docs/demo frontend (Bun or npm)
#   frontend:build   Production build of the docs/demo frontend (Bun or npm)
#   api              Run the demo Web API locally (127.0.0.1:5824)
#   bridge           Run the local bridge (127.0.0.1:5823)
#
# Env:
#   PYTHON       Python executable (auto-detected; fallback: python)
#   VENV_DIR     Virtualenv dir (default: .venv)
#   FRONTEND_DIR Docs/demo site directory (default: frontend)

if command -v python3 >/dev/null 2>&1; then
  PY_DEFAULT=python3
else
  PY_DEFAULT=python
fi
PY=${PYTHON:-$PY_DEFAULT}
VENV_DIR=${VENV_DIR:-.venv}
FRONTEND_DIR=${FRONTEND_DIR:-frontend}

info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

usage() {
  sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'
}

_activate_venv() {
  # shellcheck disable=SC1090
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
  elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Windows (Git Bash, MSYS, Cygwin)
    source "$VENV_DIR/Scripts/activate"
  else
    err "Could not find venv activation script in $VENV_DIR"
    exit 2
  fi
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtualenv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi
  _activate_venv
  python -V
}

cmd_setup() {
  ensure_venv
  info "Upgrading pip and wheel"
  pip install --upgrade pip wheel
  info "Installing package (editable) and dev deps"
  pip install -e .[dev]
}

cmd_test() {
  ensure_venv
  info "Running pytest (fast) (forcing NumPy baseline)"
  # Force baseline (numpy) implementation during tests to avoid flaky
  # results from cached/generated backends in dev environments.
  UHOP_FORCE_BASELINE=1 pytest -q
}

cmd_test_all() {
  ensure_venv
  info "Running pytest (full) (forcing NumPy baseline)"
  UHOP_FORCE_BASELINE=1 pytest -q
}

cmd_lint() {
  ensure_venv
  if command -v ruff >/dev/null 2>&1; then
    info "ruff: python lint"
    ruff check uhop tests || true
  fi
  if command -v flake8 >/dev/null 2>&1; then
    info "flake8: python style checks"
    flake8 uhop tests || true
  else
    info "flake8 not installed; skipping"
  fi
  if command -v markdownlint-cli2 >/dev/null 2>&1; then
    info "markdownlint: docs"
    markdownlint-cli2 "**/*.md" || true
  fi
}

cmd_fmt() {
  ensure_venv
  if command -v ruff >/dev/null 2>&1; then
    info "ruff: auto-fixing lint issues"
    ruff check --fix uhop tests || true
  fi
  if command -v black >/dev/null 2>&1; then
    info "black: formatting"
    black uhop tests
  fi
  if command -v isort >/dev/null 2>&1; then
    info "isort: imports"
    isort uhop tests
  fi
  if command -v prettier >/dev/null 2>&1; then
    info "prettier: formatting web"
    (cd "$FRONTEND_DIR" && prettier -w .) || true
  elif _has_bun; then
    info "prettier: formatting web via bunx"
    (cd "$FRONTEND_DIR" && bunx prettier -w .) || true
  elif _has_npm; then
    info "prettier: formatting web via npx"
    (cd "$FRONTEND_DIR" && npx --yes prettier -w .) || true
  fi

    # C-family kernels: OpenCL (.cl), CUDA (.cu/.cuh), C/C++ headers (style only)
    info "clang-format: formatting kernel sources if available"
    KERNEL_PATTERNS=("*.cl" "*.cu" "*.cuh" "*.h" "*.hpp" "*.c" "*.cpp")
    FILES=()
    if _has_git; then
      # Collect tracked files matching patterns
      for pat in "${KERNEL_PATTERNS[@]}"; do
        while IFS= read -r f; do
          FILES+=("$f")
        done < <(git ls-files "$pat" 2>/dev/null || true)
      done
    else
      # Fallback to find (may include node_modules if not careful)
      while IFS= read -r f; do FILES+=("$f"); done < <(find . -type f \( -name "*.cl" -o -name "*.cu" -o -name "*.cuh" -o -name "*.h" -o -name "*.hpp" -o -name "*.c" -o -name "*.cpp" \) 2>/dev/null | sed 's#^\./##')
    fi
    if [[ ${#FILES[@]} -gt 0 ]]; then
      CF_OK=0
      for f in "${FILES[@]}"; do
        _run_clang_format "$f" || CF_OK=1
      done
      if [[ $CF_OK -ne 0 ]]; then
        info "clang-format not found. Install LLVM clang-format or use: bunx clang-format or npx clang-format"
      fi
    fi
}

_has_bun() { command -v bun >/dev/null 2>&1; }
_has_npm() { command -v npm >/dev/null 2>&1; }
  _has_git() { command -v git >/dev/null 2>&1; }

  _run_clang_format() {
    local file="$1"
    if command -v clang-format >/dev/null 2>&1; then
      clang-format -i "$file" || true
    elif _has_bun; then
      bunx --yes clang-format -i "$file" || true
    elif command -v npx >/dev/null 2>&1; then
      npx --yes clang-format -i "$file" || true
    else
      return 1
    fi
  }

cmd_frontend_dev() {
  if _has_bun; then
    info "Installing frontend deps with Bun and starting dev server"
    (cd "$FRONTEND_DIR" && bun install && bun run dev)
  elif _has_npm; then
    info "Installing frontend deps with npm and starting dev server"
    (cd "$FRONTEND_DIR" && npm ci && npm run dev)
  else
    err "Neither Bun nor npm found. Please install one of them."
    exit 2
  fi
}

cmd_frontend_build() {
  if _has_bun; then
    info "Building frontend with Bun"
    (cd "$FRONTEND_DIR" && bun install && bun run build)
  elif _has_npm; then
    info "Building frontend with npm"
    (cd "$FRONTEND_DIR" && npm ci && npm run build)
  else
    err "Neither Bun nor npm found. Please install one of them."
    exit 2
  fi
}

cmd_api() {
  ensure_venv
  info "Starting demo API at http://127.0.0.1:5824"
  # Prefer installed console script for consistency with README
  uhop web-api --host 127.0.0.1 --port 5824
}

cmd_bridge() {
  ensure_venv
  info "Starting local bridge at http://127.0.0.1:5823"
  uhop web-bridge --port 5823
}

main() {
  local cmd=${1:-help}
  case "$cmd" in
    help|-h|--help) usage ;;
    setup) shift; cmd_setup "$@" ;;
    test) shift; cmd_test "$@" ;;
    test:all) shift; cmd_test_all "$@" ;;
    lint) shift; cmd_lint "$@" ;;
    fmt) shift; cmd_fmt "$@" ;;
    hooks) shift; if command -v pre-commit >/dev/null 2>&1; then info "Installing pre-commit hooks"; pre-commit install; else info "pre-commit not installed; pip install pre-commit to use this command"; fi ;;
    frontend:dev) shift; cmd_frontend_dev "$@" ;;
    frontend:build) shift; cmd_frontend_build "$@" ;;
    api) shift; cmd_api "$@" ;;
    bridge) shift; cmd_bridge "$@" ;;
    *) err "Unknown command: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
