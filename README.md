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
- **heuristic** (opt-in via `--model heuristic`) — mouth-movement amplitude
  (MediaPipe face mesh, band-pass filtered to speech-articulation rate)
  cross-correlated against a voice-activity-gated audio energy envelope.
  Fast, no extra install. **⚠️ Status: improved but still experimental.**
  An initial version tested outright unreliable (wrong sign on half the
  test windows); adding VAD gating, band-pass filtering, and a face-coverage
  check got 4 of 6 windows within ~600ms of ground truth — real progress,
  but one window was still off by 4.7 seconds, so it's not yet trustworthy
  as a sole source of truth (see [Testing Results](#testing-results) for
  both versions' full numbers). Prefer syncnet when available; use this
  only as a rough guess, cross-checked against multiple windows.

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

# Hands-off: scan the file for good windows and use their consensus (see below)
./vlcsync.sh fix movie.mp4 --scan --apply
```

Sign convention: positive `offset_ms` means audio is ahead of video and
should be delayed; negative means audio is behind and should be advanced.
This matches VLC's `--audio-desync` and ffmpeg's `-itsoffset` directly.

### `scan` — for when you don't want to hand-pick a window

`--start`/`--duration` require picking a window with a visible talking face
yourself. `scan` automates that: it cheaply probes many short candidate
windows spread across the file for face/voice presence (MediaPipe, no
offset math — a few seconds each), then runs the real detector only on the
best of those, in parallel, and reports the median offset plus how many
windows agreed with it.

```bash
./vlcsync.sh scan movie.mp4                    # defaults: syncnet, 6 windows, ~2-3 min on a ~40min file
./vlcsync.sh scan movie.mp4 --windows 10        # more windows = more robust median, more time
./vlcsync.sh fix movie.mp4 --scan --apply       # scan, then remux using the consensus offset
```

This exists because two more obvious approaches were tried first and didn't
hold up — see [Scan: what didn't work first](#scan-what-didnt-work-first).

## Testing Results

**Setup:** a real ~42-minute mp4 (852x480, 25fps, h264/aac) with audio
manually confirmed delayed by exactly **5000ms** (fixed in VLC by setting
Track Synchronization → Audio to -5000ms). Tested on macOS 12.6.5, Apple M1
Pro, Python 3.12 venv. Each row is an independent, non-overlapping 20-second
window (`--start N --duration 20 --max-offset 6000`); ground truth is
**-5000ms** throughout.

### heuristic — improved, still not fully reliable

**v1** (raw RMS energy vs. absolute mouth-opening ratio, no gating):

| `--start` | offset returned | correlation | face frames | vs. true -5000ms |
|---|---|---|---|---|
| 120s | +2010ms | 0.232 | 215/300 | off by 7010ms |
| 300s | +5030ms | 0.062 | 211/300 | off by 10030ms, wrong sign |
| 600s | +5730ms | 0.198 | 50/300 | off by 10730ms, wrong sign |
| 900s | 0ms | -2.0 (no face) | 0/300 | no result — face never detected |
| 1200s | -580ms | 0.243 | 235/300 | off by 4420ms |
| 1800s | +1540ms | 0.198 | 110/300 | off by 6540ms |

Not one of the six windows landed anywhere near correct, and three got the
sign wrong. Root cause: raw RMS energy reacts to the whole track (music,
ambient noise, other speakers) rather than specifically to this face's
speech, the mouth signal was a static open/closed ratio rather than
movement, nothing filtered for the ~1.5-6Hz rate speech articulation
actually happens at, and windows with sparse/interrupted face detection
were scored the same as clean ones.

**v2** (same file, same windows, after fixing all four of those): audio
gated by WebRTC voice-activity detection instead of raw RMS, both signals
band-pass filtered to 1.5-6Hz before correlating, and a face-coverage
check (≥50% of sampled frames) that downgrades confidence instead of
silently trusting sparse tracking:

| `--start` | offset returned | correlation | face coverage | voiced % | vs. true -5000ms |
|---|---|---|---|---|---|
| 120s | -4390ms | 0.267 | 86% | 86% | off by 610ms, correct sign |
| 300s | -330ms | 0.263 | 84% | 90% | off by 4670ms — still wrong |
| 600s | -3250ms | 0.221 | 20% (flagged low-confidence) | 95% | off by 1750ms |
| 900s | 0ms | -2.0 (no face) | 0% | 96% | no result — face never detected |
| 1200s | -5030ms | 0.247 | 94% | 89% | off by 30ms — essentially exact |
| 1800s | -4910ms | 0.219 | 44% (flagged low-confidence) | 95% | off by 90ms |

4 of 6 windows now land within ~600ms with the correct sign (two of them
within 100ms), and the two weak spots correctly flag themselves as
low-confidence via the new coverage check rather than confidently
returning garbage — a real improvement, not a wash. But 300s is still off
by 4.7 seconds despite "medium" confidence, which means the confidence
score still isn't fully trustworthy on its own. **Still not recommended as
a sole source of truth** — cross-checking multiple windows (or against
syncnet, when available) remains necessary, though the signal is now
usable as a genuine hint rather than noise.

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

### Scan: what didn't work first

**A single longer window, tried on the same file:** tripling `--duration`
from 20s to 60s at a window that was already working well (1200s, syncnet,
30ms off) made it *worse* — wrong sign, +5810ms instead of -5030ms — because
the extra time ran into a different scene and a single global correlation
has no way to know part of the window stopped being useful. On syncnet's
known failure window (1800s, 12 scene cuts in 20s), tripling the duration
found 32 scene cuts in 60s and still failed — that whole region is just
rapid-cut throughout, so more duration at the same starting point doesn't
help if the content itself never holds still. **Longer single windows are
not a fix for either backend.**

**Fixed, evenly-spaced windows with retries, tried next:** running syncnet
on 6 windows spread evenly across the file, sequentially, with up to 3
retries each on a nearby offset if a slot failed — worked, but slowly:
~3-18 minutes depending on how many slots needed retries, because each
syncnet window costs ~45s regardless of whether it succeeds (its own face
detector, S3FD via PyTorch, runs at only ~11fps on CPU, and that cost is
paid before the pipeline even knows whether a usable face track exists).
On this particular file, all 6 initial evenly-spaced slots happened to land
on rapid-cut stretches and needed retries. **Accurate, but too slow to be
practical.**

**Cheap probing + parallelism, what actually shipped:** MediaPipe (already
used by the heuristic backend) can answer "is there a continuously-visible
face and audible speech here" in ~5s for an 8s clip — about 9x faster than
paying for syncnet's own face detector just to find out a window is
unusable. `scan` now probes ~50-80 short candidates across the file with
MediaPipe (ignoring the heuristic's own offset answer, which isn't reliable
enough to trust — only its face-coverage/voiced-fraction numbers), keeps
the ones that clear a coverage threshold and are spread apart, oversamples
by 2x to absorb the cases where a probe passes but the real detector still
fails (they don't perfectly agree — probing is short/MediaPipe, real
detection is longer/S3FD with a strict 4-second *continuous* track
requirement), and runs the real detector on all of them concurrently
(thread-capped to avoid the parallel processes fighting each other for
CPU). Result on the same file, run three times:

| Run | Windows selected | Succeeded | Consensus offset | Time |
|---|---|---|---|---|
| 1 | 6 (clustered ~1065-1250s + one at 289s) | 4/6, agreement 3/4 | -4880ms (120ms off) | 3:17 |
| 2 (wider spacing enforced) | 5 (spread 60-2438s) | 2/5, agreement 0/2 | -4560ms (440ms off) | 2:16 |
| 3 (+ 2x oversampling) | 10 (spread 60-2438s) | 3/10, agreement 2/3 | -4880ms (120ms off) | 2:43 |

8 cores (6P+2E) on the test machine; `--jobs` defaults to `cpu_count - 1`.
Consensus landed within 120-440ms of the true -5000ms across all three
runs — accurate every time — in roughly 2-3.5 minutes instead of the
naive approach's 3-18 minute range, on a ~42-minute file. Run 2 shows why
oversampling (run 3) matters: enforcing real spread across the file is
good for independence, but it also means fewer of the passing candidates
are usable, so without oversampling you can end up with only 2 successes
and 0 agreement between them.

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
