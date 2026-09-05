#!/usr/bin/env python3
"""
Lip-vs-audio A/V sync offset detector.

Heuristic (not full SyncNet): tracks mouth-opening amplitude from face-mesh
landmarks across a video window, computes a voice-activity-gated audio
energy envelope over the same window, band-pass filters both to the ~1.5-6Hz
range where speech articulation actually lives, and finds the time shift
that maximizes their correlation.

Sign convention: a positive offset_ms means the AUDIO is ahead of the video
and should be DELAYED by that many ms to line up (matches VLC's
--audio-desync and ffmpeg's -itsoffset semantics). Negative means audio is
behind and should be advanced.

Requires a segment with a visible, talking face and audible speech. Silence,
music, off-camera narration, sparse/interrupted face detection, or a
non-speaking face on screen will produce a low confidence score or an
outright refusal -- check face/voice coverage before trusting the offset.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os

import numpy as np
import cv2
from scipy.signal import butter, filtfilt

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

try:
    import webrtcvad
except ImportError:
    print("error: webrtcvad not installed. Run: vlcsync.sh install (or setup_venv.sh)", file=sys.stderr)
    sys.exit(1)

GRID_HZ = 100.0  # 10ms resolution -- also the VAD frame size at 16kHz (webrtcvad requires 10/20/30ms)
AUDIO_SR = 16000
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291

VAD_MODE = 2  # webrtcvad aggressiveness, 0 (permissive) - 3 (strict)
BANDPASS_LOW_HZ = 1.5   # speech articulation rate lives roughly in 1.5-6Hz;
BANDPASS_HIGH_HZ = 6.0  # filtering to this band suppresses head-pose drift, camera cuts, and landmark jitter alike
MIN_FACE_FRAMES = 10
MIN_FACE_COVERAGE = 0.5  # fraction of sampled frames that must have a detected face to trust the window
MIN_VOICED_FRACTION = 0.05  # fraction of the audio window VAD must find speech in


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
    sampled_count = 0
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
            sampled_count += 1
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
    return np.array(times), np.array(values), sampled_count


def extract_audio_envelope(video_path, start, duration):
    """RMS energy envelope, gated to zero outside VAD-detected speech so
    background music/ambient noise/silence can't dominate the correlation
    the way raw loudness does."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
            "-i", video_path, "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
            "-f", "wav", wav_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")

        audio_i16, sr = sf.read(wav_path, dtype="int16")
        if audio_i16.ndim > 1:
            audio_i16 = audio_i16.mean(axis=1).astype(np.int16)

        win = max(1, int(round(sr / GRID_HZ)))  # 160 samples at 16kHz/100Hz grid == a 10ms VAD frame
        n_frames = max(1, len(audio_i16) // win)

        vad = webrtcvad.Vad(VAD_MODE)
        audio_f = audio_i16.astype(np.float32) / 32768.0

        env = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            chunk_i16 = audio_i16[i * win:(i + 1) * win]
            chunk_f = audio_f[i * win:(i + 1) * win]
            env[i] = np.sqrt(np.mean(chunk_f ** 2) + 1e-12)
            voiced[i] = vad.is_speech(chunk_i16.tobytes(), sr)

        voiced_fraction = float(np.mean(voiced)) if n_frames else 0.0
        gated_env = env * voiced
        times = start + np.arange(n_frames) * (win / sr)
        return times, gated_env, voiced_fraction
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


def bandpass(signal, fs, low_hz=BANDPASS_LOW_HZ, high_hz=BANDPASS_HIGH_HZ, order=3):
    """Isolates the speech-articulation-rate band. This also does double duty
    as a motion extractor: filtering out everything below ~1.5Hz removes
    slowly-varying bias (head pose, a static open mouth, an overall loudness
    trend) and leaves mainly the oscillation that reflects actual movement."""
    if signal.std() < 1e-9:
        return signal
    nyq = fs / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return filtfilt(b, a, signal)


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

    print(
        "warning: this heuristic backend is experimental -- validate against --model syncnet "
        "if it's installed, or cross-check multiple --start windows for agreement, before "
        "trusting its output (see README Accuracy section).",
        file=sys.stderr,
    )
    print(f"analyzing {args.video} [{args.start:.1f}s .. {args.start + args.duration:.1f}s] ...", file=sys.stderr)

    vt, vv, sampled_count = extract_mouth_signal(args.video, args.start, args.duration, args.sample_fps)
    face_frames = len(vt)
    face_coverage = (face_frames / sampled_count) if sampled_count else 0.0
    print(f"face detected in {face_frames}/{sampled_count} sampled frames ({face_coverage * 100:.0f}% coverage)",
          file=sys.stderr)

    low_face_signal = face_frames < MIN_FACE_FRAMES or face_coverage < MIN_FACE_COVERAGE
    if face_frames < MIN_FACE_FRAMES:
        print("warning: too few frames with a detected face; pick a window with a visible talking face "
              "(--start/--duration) for a reliable result", file=sys.stderr)
    elif face_coverage < MIN_FACE_COVERAGE:
        print(f"warning: face detected in only {face_coverage * 100:.0f}% of sampled frames "
              f"(need >={MIN_FACE_COVERAGE * 100:.0f}%) -- too many gaps/cuts/occlusions to trust this "
              "window; try a steadier close-up segment", file=sys.stderr)

    at, av, voiced_fraction = extract_audio_envelope(args.video, args.start, args.duration)
    print(f"voice activity detected in {voiced_fraction * 100:.0f}% of the audio window", file=sys.stderr)
    if voiced_fraction < MIN_VOICED_FRACTION:
        print("warning: almost no speech detected in this window (music/silence/noise?) -- "
              "try a segment with clearer dialogue", file=sys.stderr)

    grid_t, mouth_grid = resample_to_grid(vt if face_frames else np.array([args.start]),
                                           vv if face_frames else np.array([0.0]),
                                           args.start, args.duration)
    _, audio_grid = resample_to_grid(at, av, args.start, args.duration)

    mouth_z = zscore(bandpass(mouth_grid, GRID_HZ))
    audio_z = zscore(bandpass(audio_grid, GRID_HZ))

    max_shift_samples = int(round(args.max_offset / 1000.0 * GRID_HZ))
    shift, corr = find_best_shift(mouth_z, audio_z, max_shift_samples)
    offset_ms = shift * (1000.0 / GRID_HZ)

    confidence = "low"
    if not low_face_signal and voiced_fraction >= MIN_VOICED_FRACTION:
        if corr >= 0.5:
            confidence = "high"
        elif corr >= 0.25:
            confidence = "medium"

    report = {
        "offset_ms": round(offset_ms, 1),
        "correlation": round(corr, 3),
        "confidence": confidence,
        "face_frames_analyzed": face_frames,
        "face_sampled_frames": sampled_count,
        "face_coverage": round(face_coverage, 3),
        "voiced_fraction": round(voiced_fraction, 3),
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
