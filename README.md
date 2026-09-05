# vlc-av-sync

Video files sometimes have audio and video tracks that drift out of sync —
a fixed offset where the sound plays too early or too late relative to the
picture. VLC can compensate for this manually (Tools → Track Synchronization,
or the `--audio-desync` flag), but you have to find the right number by ear.
This tool finds that number automatically, by analyzing lip movement against
the audio track, and either launches VLC with it applied or writes a
corrected file.

```bash
./vlcsync.sh scan movie.mp4 --apply    # find the offset and write movie.synced.mp4
```

## How it works

Two independent ways to estimate the offset from a short clip of the video
(never the whole file, so multi-GB movies are fine), selected with `--model`:

- **syncnet** (default, recommended) — runs the open-source
  [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
  pipeline: detect and track a face (S3FD), crop it, and score how well its
  lip movement matches the audio at each candidate time-shift using a CNN
  trained specifically for this task. Needs a one-time extra install
  (PyTorch etc.) but is the one shown by testing to actually work — see
  [Accuracy](#accuracy).
- **heuristic** (opt-in, experimental) — a from-scratch, dependency-light
  approach: track mouth-opening amplitude with MediaPipe's face mesh,
  compute a voice-activity-gated audio energy envelope, band-pass filter
  both to the ~1.5-6Hz rate speech articulation happens at, and
  cross-correlate to find the best-matching time-shift. No trained model,
  no heavy install — but testing found it meaningfully less reliable (see
  [Accuracy](#accuracy)). Kept as a zero-install fallback, not a peer to
  syncnet.

Both need a window where a face is visible and talking — a single point in
the file isn't enough context to judge lip-sync. Rather than making you
hand-pick that window, **`scan`** automates it: it cheaply probes dozens of
short candidate clips across the file for face/voice presence using the
lightweight MediaPipe check (not the expensive syncnet pipeline), keeps the
ones that look usable, and runs the real detector on only those — several
at once, in parallel — then reports the median offset across the successful
runs. See [Performance](#performance) for why this two-stage design exists
and what it actually costs.

## Install

Three ways, pick one:

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
(`brew install --cask vlc`, or from videolan.org) if you want the `fix`
command's auto-launch/`--play` to work. Everything else above is
self-contained.

## Usage

```bash
# Hands-off: probe the file, detect on the best windows, write a corrected copy
./vlcsync.sh fix movie.mp4 --scan --apply

# Same, but just launch VLC with the found offset instead of writing a new file
./vlcsync.sh fix movie.mp4 --scan

# Scan on its own, without fixing anything yet
./vlcsync.sh scan movie.mp4

# You already know (or want to inspect) a good window
./vlcsync.sh detect movie.mp4 --model syncnet --start 300 --duration 20
./vlcsync.sh fix movie.mp4 --model syncnet --start 300 --duration 20 --apply

# Skip detection entirely, apply a known offset
./vlcsync.sh fix movie.mp4 --offset -5000

# Subtitle sync (ffsubsync, audio-based — a separate, well-established technique)
./vlcsync.sh subs movie.mp4 subs.srt
```

Sign convention: positive `offset_ms` means audio is ahead of video and
should be delayed; negative means audio is behind and should be advanced.
This matches VLC's `--audio-desync` and ffmpeg's `-itsoffset` directly, so
no sign-flipping is needed to use either.

## Performance

Single-window timings on a 2021 M1 Pro (8 cores: 6 performance + 2
efficiency), macOS 12.6.5, against a 10-second clip:

| Step | Typical time | Why |
|---|---|---|
| heuristic detection | ~1.5-2s | MediaPipe + numpy, lightweight |
| syncnet detection | ~17-25s | dominated by its own face detector (S3FD via PyTorch), which runs at only ~11fps on CPU |
| probe (used by `scan`) | ~1.5-2s | it's just a heuristic-backend call, reused for its face/voice-coverage numbers only |

Because syncnet's face detector is the expensive part *regardless of
whether the window turns out usable*, naively running it on several
evenly-spaced windows can take minutes even when most of them fail. `scan`
avoids that by probing cheaply first (see [How it works](#how-it-works))
and running the expensive part only on promising candidates, in parallel.
Measured end-to-end on the ~42-minute test file described in
[Accuracy](#accuracy), across 3 runs while the design was being tuned:

| | Time | Consensus offset |
|---|---|---|
| Naive: 6 fixed evenly-spaced windows, sequential, with retries | 3-18 min (varied run to run) | accurate when it finished, but unpredictably slow |
| `scan`: probe + parallel detection | **2:16 - 3:17** | within 120-440ms of the reference every time |

**Parallelism is real, not just nominal:** 3 identical syncnet windows took
140s run back-to-back vs. 87s run concurrently on this 8-core machine — a
genuine ~1.6x speedup (not the naive 3x you'd hope for; CPU contention
between processes eats into ideal scaling, but it's a substantial and
consistent win, which is why `scan` runs both its probe and detection
phases through a bounded thread pool by default).

**Two "obvious" further speedups were tried and rejected** because they
traded away accuracy, not just time:
- Shortening the detection window below syncnet's already-short 10s default
  (tried 6s and 8s, with `--min-track` lowered to allow it): both produced
  wrong answers that landed suspiciously close to the search boundary — a
  sign the correlation had too little audio/video to work with, not a
  real result.
- Lowering `--facedet-scale` (the frame downscale factor before face
  detection) below the upstream default of 0.25: 0.20 was ~28% faster
  *and* still correct on one test window, but completely failed to detect
  a face at all on a different window that worked fine at the default —
  a real regression, not a safe tuning knob, so the default is unchanged.

If you have spare CPU budget and want more robustness rather than more
speed, `--windows` (more consensus samples) is the parameter to raise, not
window duration or detection resolution.

## Accuracy

**A methodology caveat that matters:** the reference offset used below was
a **manual, round-number correction** (the user played the file, set VLC's
audio delay to -5000ms, and it looked right) — not a frame-accurate
verified value. Note that both `syncnet` and `scan` independently and
repeatedly converged on **~-4880ms to -4960ms**, consistently, across many
different windows and separate runs — tighter clustering than you'd expect
from noise. It's plausible the *actual* offset is closer to that ~4.9s
figure than the round -5000ms reference, i.e. **the tool may be more
precise than the manual reference it's being scored against.** The "off by
Nms" figures below should be read as "distance from a rough reference,"
not "measured error against ground truth."

### syncnet — consistent

Independent, non-overlapping 20-second windows on a real ~42-minute mp4
(852x480, 25fps, h264/aac):

| `--start` | offset returned | syncnet confidence | tracks found | vs. ~-5000ms reference |
|---|---|---|---|---|
| 120s | -4920ms | 1.852 | 2 | 80ms |
| 300s | -4960ms | 1.764 | 1 | 40ms |
| 1200s | -4880ms | 3.734 | 2 | 120ms |
| 1800s | *(no result)* | — | 0 | window had 12 scene cuts in 20s — no face track lasted the required ~4s, correctly reported as a failure rather than a guess |

All three successful windows landed within 120ms (3 frames at 25fps) of
the reference, despite each reporting only "low" or "medium" confidence by
this tool's own threshold — **cross-window agreement turned out to be a
better real-world confidence signal than either backend's own per-run
confidence score.** That's the whole rationale behind `scan`.

### heuristic — improved, still not fully reliable

**v1** (raw RMS energy vs. absolute mouth-opening ratio, no gating):

| `--start` | offset returned | correlation | face frames | vs. ~-5000ms reference |
|---|---|---|---|---|
| 120s | +2010ms | 0.232 | 215/300 | 7010ms, wrong sign |
| 300s | +5030ms | 0.062 | 211/300 | 10030ms, wrong sign |
| 600s | +5730ms | 0.198 | 50/300 | 10730ms, wrong sign |
| 900s | 0ms | -2.0 (no face) | 0/300 | no result |
| 1200s | -580ms | 0.243 | 235/300 | 4420ms |
| 1800s | +1540ms | 0.198 | 110/300 | 6540ms, wrong sign |

Not one of the six windows landed anywhere near correct, and three got the
sign wrong. Root cause: raw RMS energy reacts to the whole track (music,
ambient noise, other speakers) rather than specifically to this face's
speech; the mouth signal was a static open/closed ratio rather than
movement; nothing filtered for the ~1.5-6Hz rate speech articulation
actually happens at; and windows with sparse/interrupted face detection
were scored the same as clean ones.

**v2** (same file, same windows, after fixing all four): audio gated by
WebRTC voice-activity detection instead of raw RMS, both signals band-pass
filtered to 1.5-6Hz before correlating, and a face-coverage check (≥50% of
sampled frames) that downgrades confidence instead of silently trusting
sparse tracking:

| `--start` | offset returned | correlation | face coverage | voiced % | vs. ~-5000ms reference |
|---|---|---|---|---|---|
| 120s | -4390ms | 0.267 | 86% | 86% | 610ms, correct sign |
| 300s | -330ms | 0.263 | 84% | 90% | 4670ms — still wrong |
| 600s | -3250ms | 0.221 | 20% (flagged low-confidence) | 95% | 1750ms |
| 900s | 0ms | -2.0 (no face) | 0% | 96% | no result |
| 1200s | -5030ms | 0.247 | 94% | 89% | 30ms |
| 1800s | -4910ms | 0.219 | 44% (flagged low-confidence) | 95% | 90ms |

4 of 6 windows now land within ~600ms with the correct sign, and the two
weak spots correctly flag themselves as low-confidence via the coverage
check rather than confidently returning garbage — real improvement, not a
wash. But 300s is still off by 4.7 seconds despite "medium" confidence, so
the confidence score still isn't fully trustworthy on its own. **Not
recommended as a sole source of truth** — cross-checking multiple windows
(`scan`, or `--model syncnet` when available) remains necessary.

## Contributing / known limitations

- The heuristic backend needs real algorithmic work to be trustworthy
  standalone — see the v1→v2 root causes above for what was already fixed,
  and the fact that 300s is still wrong for what's likely left (a
  non-speaking face in frame while someone off-camera talks is one
  candidate explanation, untested).
- `scan`'s probe and syncnet's real detector don't perfectly agree on
  what counts as a "usable" window (probe checks overall face-frame
  coverage; syncnet needs one *continuous* 4-second track) — `scan`
  compensates by oversampling 2x, which works but is a coarser fix than
  reconciling the two checks directly.
- Everything here has been validated against exactly one manually-labeled
  test file. More labeled test files (ideally with a genuinely
  frame-accurate reference offset, e.g. a clap-board/tone-burst sync test
  pattern rather than a manual VLC correction) would substantially
  strengthen the numbers above. PRs adding those, or testing on Linux/
  Windows where the macOS-specific patches below shouldn't be needed, are
  welcome.

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
original code already worked — if you test there and these turn out to be
unnecessary (or something else breaks instead), that's useful information,
please open an issue.

Separately, `mediapipe`'s current PyPI releases (0.10.14+ and 1.0.x) ship a
macOS arm64 wheel that fails to load at all on macOS <13
(`Symbol not found: _objc_claimAutoreleasedReturnValue` — a packaging bug,
not a version-range issue). `requirements.txt` pins `mediapipe==0.10.13`,
the newest release confirmed to load correctly there. If you're on macOS
13+, feel free to bump it.
