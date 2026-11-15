#!/usr/bin/env bash
# Install PHS Studio from a source checkout.
#
#   ./scripts/install.sh            # GUI + IO + traditional ML
#   ./scripts/install.sh --dl       # also install PyTorch (CPU build recommended)
#   ./scripts/install.sh --with-qt  # apt-get the Qt runtime libs (Debian/Ubuntu)
#
# Creates .venv/ in the repository root and installs the project in editable
# mode. Re-running is safe: the virtualenv is reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WITH_DL=0
WITH_QT=0

for arg in "$@"; do
  case "$arg" in
    --dl) WITH_DL=1 ;;
    --with-qt) WITH_QT=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("PHS Studio requires Python 3.11+")
PY

if [[ "$WITH_QT" == "1" ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "warning: --with-qt only supports apt-based systems; skipping" >&2
  else
    sudo apt-get update
    sudo apt-get install -y \
      libegl1 libgl1 libglib2.0-0 libdbus-1-3 libxkbcommon-x11-0 \
      libfontconfig1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
      libxcb-xinerama0 libxcb-xkb1
  fi
fi

cd "$ROOT"
if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip

if [[ "$WITH_DL" == "1" ]]; then
  pip install -e ".[dl]"
else
  pip install -e "."
fi

cat <<EOF

Installed in $ROOT/.venv

  source .venv/bin/activate
  phs

Headless check:  QT_QPA_PLATFORM=offscreen phs
Demo project:    python scripts/make_demo_project.py --out /workspace/phs_demo
EOF
