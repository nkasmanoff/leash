#!/usr/bin/env bash
# setup_leash_demo.sh
# Bootstraps (or resets) the Leash reward-hack demo project at ~/Desktop/leash-demo.
#
# Defaults to the `discount` scenario (Acme storefront pricing). Pass another
# scenario name as $1 to use a different one (run `python -m scripts.scenarios list`
# to see available scenarios).
#
# Usage:
#   bash scripts/setup_leash_demo.sh                  # discount scenario
#   bash scripts/setup_leash_demo.sh business_days    # date-math scenario
#   SCENARIO=stats bash scripts/setup_leash_demo.sh   # stats scenario via env

set -euo pipefail

SCENARIO="${1:-${SCENARIO:-discount}}"
DEMO_DIR="${HOME}/Desktop/leash-demo"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO}"
python -m scripts.scenarios bootstrap --force "${SCENARIO}" "${DEMO_DIR}"
