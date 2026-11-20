#!/bin/bash
# Launch 360 Views (macOS / Linux). First run sets up a private Python environment in .venv.
#   ./start.sh                     open the app (reopens your last project)
#   ./start.sh VID_….insv          open / create a project for that recording
#   ./start.sh doctor              check that ffmpeg etc. are installed
set -e
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "360 Views needs Python 3.9 or newer."
  if [ "$(uname)" = "Darwin" ]; then echo "  Install it with:  brew install python   (or from https://www.python.org/downloads/)"; else echo "  Install it with your package manager, e.g.  sudo apt install python3 python3-venv"; fi
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (one time)…"
  "$PY" -m venv .venv || { echo "Couldn't create a Python environment (on Debian/Ubuntu: sudo apt install python3-venv)"; exit 1; }
fi
STAMP=.venv/.requirements
if ! cmp -s requirements.txt "$STAMP"; then
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
  cp requirements.txt "$STAMP"
  .venv/bin/python -m views360 doctor || true
fi

if ! command -v ffmpeg >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/ffmpeg ] && [ ! -x /usr/local/bin/ffmpeg ] && [ -z "$VIEWS360_FFMPEG" ]; then
  echo "360 Views needs ffmpeg."
  if [ "$(uname)" = "Darwin" ]; then echo "  Install it with:  brew install ffmpeg   (Homebrew: https://brew.sh)"; else echo "  Install it with:  sudo apt install ffmpeg   (or your distro's package manager)"; fi
  exit 1
fi

exec .venv/bin/python -m views360 "$@"
