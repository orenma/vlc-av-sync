# vlc-av-sync

Detect and fix audio/video sync offsets in video files, using either a
lightweight heuristic or an open-source lip-sync model, and hand the result
to VLC (`--audio-desync`) or write a corrected file with `ffmpeg`.

Two detection backends, selected with `--model`:

- **syncnet** (default) — the open-source [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
  pipeline (S3FD face detection + tracking, then a CNN trained specifically
  for audio/video sync scoring). Heavier (PyTorch etc., one-time extra
  install — run `install-syncnet` / `setup_venv.sh --with-syncnet` first),
  but reliably found a known 5-second offset within ~120ms, consistently
  across independent windows of the same file. If it isn't installed yet,
  `detect`/`fix` fail loudly with install instructions rather than quietly
  falling back to the unreliable option below.
- **heuristic** (opt-in via `--model heuristic`) — mouth-opening amplitude
  (MediaPipe face mesh) cross-correlated against the audio's RMS energy
  envelope. Fast, no extra install. **⚠️ Status: unreliable, not
  recommended.** Tested against a known offset it returned scattered,
  mostly-wrong results (see [Testing Results](#testing-results)) — it needs
  real work (a better signal than RMS energy, at minimum) before it should
  be trusted. Kept only as a zero-install fallback, printed with a warning
  every time it runs.

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

## Testing Results

**Setup:** a real ~42-minute mp4 (852x480, 25fps, h264/aac) with audio
manually confirmed delayed by exactly **5000ms** (fixed in VLC by setting
Track Synchronization → Audio to -5000ms). Tested on macOS 12.6.5, Apple M1
Pro, Python 3.12 venv. Each row is an independent, non-overlapping 20-second
window (`--start N --duration 20 --max-offset 6000`); ground truth is
**-5000ms** throughout.

### heuristic — unreliable

| `--start` | offset returned | correlation | face frames | vs. true -5000ms |
|---|---|---|---|---|
| 120s | +2010ms | 0.232 | 215/300 | off by 7010ms |
| 300s | +5030ms | 0.062 | 211/300 | off by 10030ms, wrong sign |
| 600s | +5730ms | 0.198 | 50/300 | off by 10730ms, wrong sign |
| 900s | 0ms | -2.0 (no face) | 0/300 | no result — face never detected |
| 1200s | -580ms | 0.243 | 235/300 | off by 4420ms |
| 1800s | +1540ms | 0.198 | 110/300 | off by 6540ms |

Not one of the six windows landed anywhere near correct, and three got the
sign wrong. The RMS-energy-vs-mouth-amplitude correlation this backend
relies on just isn't a strong enough signal on real dialogue — mouth
movement and audio loudness correlate loosely at best, and this file's
noise floor (music, ambient sound, multiple speakers) buries it. This
backend needs real algorithmic work, not a parameter tweak, before it
should be trusted for anything beyond a rough first guess.

### syncnet — consistent and accurate

| `--start` | offset returned | syncnet confidence | tracks found | vs. true -5000ms |
|---|---|---|---|---|
| 120s | -4920ms | 1.852 | 2 | off by 80ms |
| 300s | -4960ms | 1.764 | 1 | off by 40ms |
| 1200s | -4880ms | 3.734 | 2 | off by 120ms |
| 1800s | *(no result)* | — | 0 | window had 12 scene cuts in 20s — no face track lasted the required ~4s, correctly reported as a failure rather than a guess |

All three successful windows landed within 120ms (3 frames at 25fps) of the
true offset, despite each reporting only "low" or "medium" confidence by
this tool's own threshold. **Cross-checking 2-3 non-overlapping windows and
looking for agreement across them was a better real-world confidence signal
here than the raw confidence number either backend reports** — treat a
single run's confidence label as a hint, not a verdict.

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
