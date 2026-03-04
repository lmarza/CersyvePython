#!/usr/bin/env bash
set -euo pipefail

# Navigate to repo root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ABC_DIR="third_party/alpha-beta-CROWN"

# Clone alpha-beta-CROWN if the directory is empty or missing
if [ ! -f "$ABC_DIR/setup.py" ] && [ ! -f "$ABC_DIR/pyproject.toml" ]; then
    echo "alpha-beta-CROWN not found, cloning..."
    rm -rf "$ABC_DIR"
    git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git "$ABC_DIR"
fi

# Install alpha-beta-CROWN deps into the current environment
# 1) auto_LiRPA
pip install -e "$ABC_DIR/auto_LiRPA"

# 2) abcrown + extra (skip gurobipy which requires Python >=3.9)
grep -v 'gurobipy' "$ABC_DIR/complete_verifier/requirements.txt" | pip install -r /dev/stdin

# 3) Install the abcrown package itself (--no-deps to avoid gurobipy)
pip install --no-deps -e "$ABC_DIR"

echo "alpha-beta-CROWN installed into current environment."
