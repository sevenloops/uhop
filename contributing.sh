#!/usr/bin/env bash
set -euo pipefail

# Contributing helper for UHOP
#
# Usage:
#   ./contributing.sh [command]
#
# Commands:
#   help            Show this help
#   setup           Create venv and install dependencies (dev)
#   test            Run Python unit tests (fast)
#   test:all        Run full Python tests (including benchmarks if enabled)
#   lint            Run basic linters/checks (flake8 if available, markdownlint-cli2 if installed)
#   fmt             Run formatters (black/isort/prettier if available)
#   frontend:dev    Install and build the docs/demo frontend for dev
#   frontend:build  Production build of the docs/demo frontend
#   api             Run the demo Web API locally (127.0.0.1:5824)
#   bridge          Run the local bridge (127.0.0.1:5823)
#
# Env:
#   PYTHON       Python executable (default: python3)
#   VENV_DIR     Virtualenv dir (default: .venv)
#   FRONTEND_DIR Docs/demo site directory (default: frontend)

PY=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
FRONTEND_DIR=${FRONTEND_DIR:-frontend}

info() { echo -e "\033[1;34m[INFO]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

usage() {
  sed -n '1,40p' "$0" | sed 's/^# \{0,1\}//'
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtualenv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
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
  UHOP_FORCE_BASELINE=1 pytest -q
}

cmd_test_all() {
  ensure_venv
  info "Running pytest (full) (forcing NumPy baseline)"
  UHOP_FORCE_BASELINE=1 pytest -q
}

cmd_lint() {
  ensure_venv
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
  fi
}

cmd_frontend_dev() {
  info "Installing frontend deps (npm ci)"
  (cd "$FRONTEND_DIR" && npm ci && npm run dev)
}

cmd_frontend_build() {
  info "Building frontend (npm ci && build)"
  (cd "$FRONTEND_DIR" && npm ci && npm run build)
}

cmd_api() {
  ensure_venv
  info "Starting demo API at http://127.0.0.1:5824"
  python -m uhop.web_api --host 127.0.0.1 --port 5824
}

cmd_bridge() {
  ensure_venv
  info "Starting local bridge at http://127.0.0.1:5823"
  python -m uhop.web_bridge --port 5823
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
    frontend:dev) shift; cmd_frontend_dev "$@" ;;
    frontend:build) shift; cmd_frontend_build "$@" ;;
    api) shift; cmd_api "$@" ;;
    bridge) shift; cmd_bridge "$@" ;;
    *) err "Unknown command: $cmd"; usage; exit 1 ;;
  esac
}

main "$@"
