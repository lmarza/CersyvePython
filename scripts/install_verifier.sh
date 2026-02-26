#!/usr/bin/env bash
set -euo pipefail

git submodule update --init --recursive

# Install alpha-beta-CROWN deps into the current environment
# 1) auto_LiRPA
pip install -e third_party/alpha-beta-CROWN/auto_LiRPA

# 2) abcrown + extra
pip install -r third_party/alpha-beta-CROWN/complete_verifier/requirements.txt

echo "alpha-beta-CROWN installed into current environment."
