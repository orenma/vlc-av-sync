#!/usr/bin/env bash
# setup_venv.sh [--with-syncnet]
#
# Pip-only, brew-free setup: creates .venv and installs everything
# (including a static ffmpeg binary via imageio-ffmpeg) into it.
# vlcsync.sh auto-detects .venv and uses it once this has run.
#
# --with-syncnet also installs the heavier open-source SyncNet backend
# (torch etc., ~1-2GB) and downloads its pretrained weights.
#
# Set VLCSYNC_PYTHON to pick the interpreter used to create the venv
# (default: python3). Matters in practice: on a brand-new Python release,
# some native-extension wheels (mediapipe in particular) may not have
# stable builds yet -- if you hit a dlopen/symbol error from mediapipe,
# rerun with an older interpreter, e.g.:
#   VLCSYNC_PYTHON=python3.12 ./setup_venv.sh
#
# Not covered here: VLC.app itself. It's a GUI application, not a Python
# package, so it still needs `brew install --cask vlc` or a manual download
# from videolan.org for the `fix` command's auto-launch/--play to work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON_BIN="${VLCSYNC_PYTHON:-python3}"

WITH_SYNCNET=false
if [ "${1:-}" = "--with-syncnet" ]; then
  WITH_SYNCNET=true
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "==> creating venv at $VENV_DIR (using $PYTHON_BIN, $($PYTHON_BIN --version 2>&1))"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> pip install -r requirements.txt"
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# imageio-ffmpeg bundles a static ffmpeg binary; expose it as a plain
# `ffmpeg` on PATH inside the venv so vlcsync.sh and the detector scripts
# (which just shell out to `ffmpeg`) work unchanged, with no brew/apt needed.
FFMPEG_BIN="$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
ln -sf "$FFMPEG_BIN" "$VENV_DIR/bin/ffmpeg"
echo "==> ffmpeg (via imageio-ffmpeg) linked at $VENV_DIR/bin/ffmpeg"

if $WITH_SYNCNET; then
  echo "==> pip install -r requirements-syncnet.txt (torch etc., this is the big one)"
  pip install -r "$SCRIPT_DIR/requirements-syncnet.txt"

  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/lib_syncnet_install.sh"
  clone_and_fetch_syncnet_weights "$SCRIPT_DIR/third_party/syncnet_python"
fi

deactivate

cat <<EOF

done. This venv is fully pip-based -- no brew needed for ffmpeg, mediapipe,
$($WITH_SYNCNET && echo "torch/SyncNet, ")or any other Python dependency.

vlcsync.sh auto-detects .venv and uses it automatically. Just run, e.g.:
  ./vlcsync.sh detect movie.mp4 --start 60 --duration 20
$($WITH_SYNCNET && echo "  ./vlcsync.sh detect movie.mp4 --model syncnet --start 60 --duration 20")

Note: VLC.app itself is still a separate install (brew install --cask vlc,
or download from videolan.org) -- it's a GUI app, not something pip can
provide. Everything else (detection, ffmpeg remuxing, ffsubsync) is
self-contained in .venv.
EOF
