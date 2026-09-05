#!/usr/bin/env bash
# vlcsync.sh -- install deps, auto-detect A/V offset via lip/audio correlation,
# and either fix the file or launch VLC with the correction applied.
#
# Subcommands:
#   vlcsync.sh install
#   vlcsync.sh install-syncnet
#   vlcsync.sh detect  <video> [--model heuristic|syncnet] [--start S] [--duration S] [--max-offset MS]
#   vlcsync.sh fix     <video> [--offset MS | (detect options)] [--apply] [--play]
#   vlcsync.sh subs    <video> <subtitle.srt> [-o output.srt]
#
# Two detection backends, pick with --model:
#   syncnet (default) -- open-source joonson/syncnet_python (S3FD face
#     detection + tracking, then a trained two-stream CNN). Heavier
#     (PyTorch etc.) -- run `install-syncnet` once first -- but consistently
#     landed within ~120ms of the true offset across independent windows in
#     testing. If it isn't installed yet, detect/fix fail loudly with
#     install instructions rather than silently using the fallback below.
#   heuristic -- mouth-movement (band-pass filtered) vs. VAD-gated audio
#     energy correlation (MediaPipe + numpy + webrtcvad). Fast, no extra
#     install. IMPROVED BUT STILL EXPERIMENTAL: 4 of 6 test windows landed
#     within ~600ms of a known 5000ms offset after adding VAD gating,
#     band-pass filtering, and a face-coverage check, but one window was
#     still off by 4.7s (see README Testing Results). Pass --model
#     heuristic explicitly to use it, and cross-check multiple windows --
#     don't trust a single run's confidence score alone.
# Both backends analyze a short window, not the whole file, so large
# 200-400MB movies are fine -- pick --start/--duration around a clear
# dialogue scene.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECT_PY="$SCRIPT_DIR/av_sync_detect.py"
SYNCNET_DETECT_PY="$SCRIPT_DIR/syncnet_detect.py"
SYNCNET_DIR="$SCRIPT_DIR/third_party/syncnet_python"
VENV_DIR="$SCRIPT_DIR/.venv"

# If ./setup_venv.sh has been run, prefer its python3 and its ffmpeg (a
# static binary shimmed in via imageio-ffmpeg) over anything on the system.
if [ -d "$VENV_DIR" ]; then
  PY="$VENV_DIR/bin/python3"
  export PATH="$VENV_DIR/bin:$PATH"
else
  PY=python3
fi

# Launches VLC via macOS `open` if available; returns 1 (does nothing else)
# in headless environments like Docker/Linux, where there's no VLC and no
# display -- the caller is expected to print an actionable fallback.
launch_vlc() {
  if command -v open >/dev/null 2>&1; then
    open -a VLC "$@"
    return 0
  fi
  return 1
}

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: '$1' not found. Run: $0 install" >&2; exit 1; }
}

cmd_install() {
  if ! command -v brew >/dev/null 2>&1; then
    echo "error: Homebrew not found. Install it from https://brew.sh first." >&2
    echo "(Or use ./setup_venv.sh instead -- a pip-only, brew-free install into .venv.)" >&2
    exit 1
  fi

  echo "==> brew: ffmpeg, python"
  brew install ffmpeg python

  echo "==> pip: opencv-contrib-python-headless mediapipe numpy scipy soundfile ffsubsync"
  "$PY" -m pip install --user --upgrade pip
  "$PY" -m pip install --user opencv-contrib-python-headless mediapipe numpy scipy soundfile ffsubsync

  if ! [ -d "/Applications/VLC.app" ]; then
    echo "==> brew: vlc (cask)"
    brew install --cask vlc
  else
    echo "==> VLC already installed"
  fi

  echo "done. (Want the more accurate open-source model too? Run: $0 install-syncnet)"
  echo "(Prefer a clean venv instead of touching system/brew python? Run: ./setup_venv.sh)"
}

cmd_install_syncnet() {
  echo "==> this installs the open-source joonson/syncnet_python model:"
  echo "    - clones https://github.com/joonson/syncnet_python"
  echo "    - installs torch, torchvision, torchaudio, scenedetect, python_speech_features (~1-2GB)"
  echo "    - downloads pretrained weights from robots.ox.ac.uk (Oxford VGG's official SyncNet release):"
  echo "        data/syncnet_v2.model, detectors/s3fd/weights/sfd_face.pth"
  echo

  echo "==> pip: torch torchvision torchaudio scenedetect==0.6.7.1 python_speech_features tqdm"
  "$PY" -m pip install --user torch torchvision torchaudio
  "$PY" -m pip install --user "scenedetect==0.6.7.1" python_speech_features tqdm opencv-contrib-python-headless numpy scipy

  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/lib_syncnet_install.sh"
  clone_and_fetch_syncnet_weights "$SYNCNET_DIR"

  echo "done. Use: $0 detect <video> --model syncnet"
}

# Dispatches to the chosen detection backend. Recognizes --model
# heuristic|syncnet (default syncnet -- it's the one that's actually
# reliable; see README Testing Results). If syncnet isn't installed yet,
# this fails loudly with instructions rather than silently falling back to
# the unreliable heuristic. Pass --model heuristic explicitly to opt into
# the zero-install fallback anyway. Everything else is passed through to
# that backend's script.
_run_detect_raw() {
  local video="$1"; shift
  local model="syncnet"
  local rest=()

  while [ $# -gt 0 ]; do
    case "$1" in
      --model) model="$2"; shift 2 ;;
      *) rest+=("$1"); shift ;;
    esac
  done

  case "$model" in
    heuristic) "$PY" "$DETECT_PY" "$video" "${rest[@]}" ;;
    syncnet) "$PY" "$SYNCNET_DETECT_PY" "$video" "${rest[@]}" ;;
    *) echo "unknown --model: $model (use heuristic|syncnet)" >&2; exit 1 ;;
  esac
}

cmd_detect() {
  [ $# -ge 1 ] || { echo "usage: $0 detect <video> [--model heuristic|syncnet] [--start S] [--duration S] [--max-offset MS]" >&2; exit 1; }
  require_cmd ffmpeg
  _run_detect_raw "$@"
}

# Runs detect and returns just the numeric offset_ms on stdout, letting all
# diagnostics pass through to stderr live.
detect_offset_ms() {
  local video="$1"; shift
  _run_detect_raw "$video" "$@" 2>&2 | tail -n1
}

cmd_fix() {
  [ $# -ge 1 ] || { echo "usage: $0 fix <video> [--offset MS] [--apply] [--play] [detect options]" >&2; exit 1; }
  local video="$1"; shift

  local offset=""
  local apply=false
  local play=false
  local detect_args=()

  while [ $# -gt 0 ]; do
    case "$1" in
      --offset) offset="$2"; shift 2 ;;
      --apply) apply=true; shift ;;
      --play) play=true; shift ;;
      --*) detect_args+=("$1" "$2"); shift 2 ;;
      *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
  done

  if [ -z "$offset" ]; then
    require_cmd ffmpeg
    echo "==> no --offset given, running auto-detection..." >&2
    offset="$(detect_offset_ms "$video" "${detect_args[@]}")"
    echo "==> detected offset: ${offset} ms" >&2
  fi

  if $apply; then
    require_cmd ffmpeg
    local base="${video%.*}"
    local ext="${video##*.}"
    local out="${base}.synced.${ext}"
    local offset_sec
    offset_sec="$("$PY" -c "print(${offset}/1000.0)")"
    echo "==> writing corrected file: $out (audio itsoffset=${offset_sec}s)" >&2
    ffmpeg -y -i "$video" -itsoffset "$offset_sec" -i "$video" \
      -map 0:v -map 1:a -map 0:s? -c copy "$out"
    echo "$out"
    if $play && ! launch_vlc "$out"; then
      echo "==> no VLC launcher here (e.g. Docker/Linux) -- open manually on your Mac: $out" >&2
    fi
  else
    if ! launch_vlc --args "$video" "--audio-desync=${offset%.*}"; then
      echo "==> no VLC launcher here (e.g. Docker/Linux). On your Mac, run:" >&2
      echo "    open -a VLC --args \"$video\" \"--audio-desync=${offset%.*}\"" >&2
      echo "    (or in VLC: Tools > Track Synchronization > Audio track synchronization = ${offset} ms)" >&2
    fi
  fi
}

cmd_subs() {
  require_cmd ffsubsync
  [ $# -ge 2 ] || { echo "usage: $0 subs <video> <subtitle.srt> [-o output.srt]" >&2; exit 1; }
  local video="$1" sub="$2"; shift 2
  local out="${sub%.*}.synced.srt"
  local extra_out=()
  if [ "${1:-}" = "-o" ]; then
    out="$2"
    shift 2
  fi
  ffsubsync "$video" -i "$sub" -o "$out"
  echo "$out"
}

main() {
  [ $# -ge 1 ] || usage
  local sub="$1"; shift
  case "$sub" in
    install) cmd_install "$@" ;;
    install-syncnet) cmd_install_syncnet "$@" ;;
    detect) cmd_detect "$@" ;;
    fix) cmd_fix "$@" ;;
    subs) cmd_subs "$@" ;;
    -h|--help) usage ;;
    *) echo "unknown subcommand: $sub" >&2; usage ;;
  esac
}

main "$@"
