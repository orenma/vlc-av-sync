# vlc-av-sync

Detect and fix audio/video sync offsets in video files, using either a
lightweight heuristic or an open-source lip-sync model, and hand the result
to VLC (`--audio-desync`) or write a corrected file with `ffmpeg`.

Two detection backends, selected with `--model`:

- **heuristic** (default) — mouth-opening amplitude (MediaPipe face mesh)
  cross-correlated against the audio's RMS energy envelope. Fast, no extra
  install. In practice this is noisy on real content — treat it as a quick
  first guess, not a reliable answer (see [Accuracy](#accuracy) below).
- **syncnet** — the open-source [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
  pipeline (S3FD face detection + tracking, then a CNN trained specifically
  for audio/video sync scoring). Heavier (PyTorch etc.), but in testing
  reliably found a known 5-second offset within ~100ms, consistently across
  independent windows of the same file. Needs a one-time extra install.

Both backends analyze a short window (`--start`/`--duration`) you pick
around a clear dialogue scene — never the whole file — so large movie files
are fine.

## Setup

Two ways to install, pick one:

**Pip-only venv** (no brew/system changes beyond Python itself):

```bash
./setup_venv.sh                  # heuristic backend only
./setup_venv.sh --with-syncnet   # + the syncnet backend (~1-2GB, torch etc.)
```

`vlcsync.sh` auto-detects `.venv` and uses it once this has run.

**System install via brew:**

```bash
./vlcsync.sh install              # ffmpeg, python, mediapipe, VLC.app
./vlcsync.sh install-syncnet      # + the syncnet backend
```

**Docker** (fully isolated; can detect/fix but can't launch VLC — no
display in a container):

```bash
docker build -t vlcsync .
docker run --rm -v "$PWD":/data vlcsync detect /data/movie.mp4 --model syncnet --start 300 --duration 20
```

VLC.app itself is a GUI application, not a package — install it separately
(`brew install --cask vlc` or from videolan.org) if you want the `fix`
command's auto-launch/`--play` to work.

## Usage

```bash
# Just detect the offset
./vlcsync.sh detect movie.mp4 --model syncnet --start 300 --duration 20

# Detect, then launch VLC with --audio-desync set to the detected offset
./vlcsync.sh fix movie.mp4 --model syncnet --start 300 --duration 20

# Skip detection, use a known offset
./vlcsync.sh fix movie.mp4 --offset -5000

# Remux into movie.synced.mp4 (stream-copy, no re-encode) instead of launching VLC
./vlcsync.sh fix movie.mp4 --model syncnet --start 300 --duration 20 --apply

# Subtitle sync (ffsubsync, audio-based)
./vlcsync.sh subs movie.mp4 subs.srt
```

Sign convention: positive `offset_ms` means audio is ahead of video and
should be delayed; negative means audio is behind and should be advanced.
This matches VLC's `--audio-desync` and ffmpeg's `-itsoffset` directly.

## Accuracy

Tested against a real file with a known, manually-verified 5-second audio
delay:

| Backend | 3 independent 20s windows | Verdict |
|---|---|---|
| heuristic | +2010ms, +5030ms, -580ms | Unreliable — scattered, no clear signal |
| syncnet | -4920ms, -4960ms, -4880ms | Consistent, within ~120ms of the true -5000ms |

Cross-checking 2-3 non-overlapping windows and looking for agreement is a
better confidence signal than the raw confidence number either backend
reports.

## macOS-specific fixes baked in

Getting the syncnet backend working on macOS required patching around a
platform gap: PyPI's `opencv` wheel on macOS ships **AVFoundation as its
only video backend, with no bundled FFmpeg** (unlike Linux/Windows wheels,
and unlike the conda build the upstream repo's own `environment.yml`
assumes). Two call sites in `syncnet_python` broke as a result:

- `scene_detect()` opens an intermediate `.avi` directly via OpenCV —
  AVFoundation can't decode it. Patched to hand PySceneDetect a throwaway
  H.264 transcode instead.
- `crop_video()` writes face-track crops via `cv2.VideoWriter` with the
  `XVID` fourcc into `.avi` — this silently fails to open for writing on
  macOS. Patched to `avc1`/`.mp4`, which AVFoundation can actually write.

Both patches are applied automatically and idempotently by
`lib_syncnet_install.sh` right after cloning, so a fresh install doesn't
need any manual intervention. They're harmless on Linux/Windows, where the
original code already worked.

Separately, `mediapipe`'s current PyPI releases (0.10.14+ and 1.0.x) ship a
macOS arm64 wheel that fails to load at all on macOS <13
(`Symbol not found: _objc_claimAutoreleasedReturnValue` — a packaging bug,
not a version-range issue). `requirements.txt` pins `mediapipe==0.10.13`,
the newest release confirmed to load correctly there. If you're on macOS
13+, feel free to bump it.
