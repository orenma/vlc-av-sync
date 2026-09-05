#!/usr/bin/env python3
"""
Lip-vs-audio A/V sync offset detector.

Heuristic (not full SyncNet): tracks mouth-opening amplitude from face-mesh
landmarks across a video window, computes the audio RMS energy envelope over
the same window, resamples both to a common grid, and finds the time shift
that maximizes their correlation. Speech naturally correlates mouth opening
with loudness, so the best-correlating shift approximates the audio/video
offset.

Sign convention: a positive offset_ms means the AUDIO is ahead of the video
and should be DELAYED by that many ms to line up (matches VLC's
--audio-desync and ffmpeg's -itsoffset semantics). Negative means audio is
behind and should be advanced.

Requires a segment with a visible, talking face and audible speech. Silence,
music, off-camera narration, or no face in frame will produce a low
confidence score -- check it before trusting the offset.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os

import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    print("error: mediapipe not installed. Run: vlcsync.sh install (or setup_venv.sh)", file=sys.stderr)
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("error: soundfile not installed. Run: vlcsync.sh install (or setup_venv.sh)", file=sys.stderr)
    sys.exit(1)

GRID_HZ = 100.0  # 10ms resolution for the shared correlation grid
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291


def extract_mouth_signal(video_path, start, duration, sample_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    times, values = [], []
    frame_interval = max(1, round(src_fps / sample_fps))
    frame_idx = 0
    end_ms = (start + duration) * 1000.0

    while True:
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms and pos_ms > end_ms:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            t = pos_ms / 1000.0 if pos_ms else (start + frame_idx / src_fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                upper = np.array([lm[UPPER_LIP].x * w, lm[UPPER_LIP].y * h])
                lower = np.array([lm[LOWER_LIP].x * w, lm[LOWER_LIP].y * h])
                left = np.array([lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h])
                right = np.array([lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h])
                mouth_width = np.linalg.norm(right - left)
                if mouth_width > 1e-3:
                    ratio = np.linalg.norm(lower - upper) / mouth_width
                    times.append(t)
                    values.append(ratio)
        frame_idx += 1

    cap.release()
    face_mesh.close()
    return np.array(times), np.array(values)


def extract_audio_envelope(video_path, start, duration):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
            "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", wav_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")

        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        win = max(1, int(sr / GRID_HZ))
        n_frames = max(1, len(audio) // win)
        env = np.array([
            np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2) + 1e-12)
            for i in range(n_frames)
        ])
        times = start + np.arange(n_frames) * (win / sr)
        return times, env
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def resample_to_grid(times, values, start, duration):
    grid_t = np.arange(0, duration, 1.0 / GRID_HZ)
    if len(times) < 2:
        return grid_t, np.zeros_like(grid_t)
    rel_times = times - start
    interp = np.interp(grid_t, rel_times, values, left=values[0], right=values[-1])
    return grid_t, interp


def zscore(x):
    std = x.std()
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def find_best_shift(mouth, audio, max_shift_samples):
    best_shift, best_corr = 0, -2.0
    n = min(len(mouth), len(audio))
    min_overlap = max(20, n // 4)  # require a reasonable overlap window

    for shift in range(-max_shift_samples, max_shift_samples + 1):
        if shift >= 0:
            a = mouth[shift:n]
            b = audio[: n - shift]
        else:
            a = mouth[: n + shift]
            b = audio[-shift: n]
        if len(a) < min_overlap:
            continue
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_corr, best_shift = corr, shift

    return best_shift, best_corr


def main():
    p = argparse.ArgumentParser(description="Detect audio/video sync offset via lip-vs-audio correlation.")
    p.add_argument("video")
    p.add_argument("--start", type=float, default=0.0, help="Analysis window start, seconds (default: 0)")
    p.add_argument("--duration", type=float, default=20.0, help="Analysis window length, seconds (default: 20)")
    p.add_argument("--max-offset", type=float, default=2000.0, help="Max offset to search, ms (default: 2000)")
    p.add_argument("--sample-fps", type=float, default=15.0, help="Video frames/sec to analyze (default: 15)")
    p.add_argument("--json", action="store_true", help="Print full JSON report to stdout instead of just the offset")
    args = p.parse_args()

    print(f"analyzing {args.video} [{args.start:.1f}s .. {args.start + args.duration:.1f}s] ...", file=sys.stderr)

    vt, vv = extract_mouth_signal(args.video, args.start, args.duration, args.sample_fps)
    face_frames = len(vt)
    print(f"face detected in {face_frames} sampled frames", file=sys.stderr)
    if face_frames < 10:
        print("warning: too few frames with a detected face; pick a window with a visible talking face "
              "(--start/--duration) for a reliable result", file=sys.stderr)

    at, av = extract_audio_envelope(args.video, args.start, args.duration)

    grid_t, mouth_grid = resample_to_grid(vt if face_frames else np.array([args.start]),
                                           vv if face_frames else np.array([0.0]),
                                           args.start, args.duration)
    _, audio_grid = resample_to_grid(at, av, args.start, args.duration)

    mouth_z = zscore(mouth_grid)
    audio_z = zscore(audio_grid)

    max_shift_samples = int(round(args.max_offset / 1000.0 * GRID_HZ))
    shift, corr = find_best_shift(mouth_z, audio_z, max_shift_samples)
    offset_ms = shift * (1000.0 / GRID_HZ)

    confidence = "low"
    if face_frames >= 10:
        if corr >= 0.5:
            confidence = "high"
        elif corr >= 0.25:
            confidence = "medium"

    report = {
        "offset_ms": round(offset_ms, 1),
        "correlation": round(corr, 3),
        "confidence": confidence,
        "face_frames_analyzed": face_frames,
        "window_start_s": args.start,
        "window_duration_s": args.duration,
        "note": "positive offset_ms = delay audio; negative = advance audio",
    }

    print(json.dumps(report, indent=2), file=sys.stderr)
    if confidence == "low":
        print("warning: low confidence -- try a different --start window with clearer dialogue/close-up face",
              file=sys.stderr)

    if args.json:
        print(json.dumps(report))
    else:
        print(report["offset_ms"])


if __name__ == "__main__":
    main()
