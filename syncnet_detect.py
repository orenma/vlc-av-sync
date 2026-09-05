#!/usr/bin/env python3
"""
SyncNet-based A/V offset detector -- wraps the open-source
joonson/syncnet_python pipeline (S3FD face detection + face tracking, then a
trained two-stream CNN that scores audio/video sync per candidate shift).

Heavier and more accurate than av_sync_detect.py's correlation heuristic:
real face detection (works with off-center/partial faces), a model trained
specifically for this task, and a proper confidence score. Requires
`vlcsync.sh install-syncnet` first (clones the repo, installs torch etc.,
downloads the pretrained weights).

Sign convention matches av_sync_detect.py: positive offset_ms = audio is
ahead of video and should be delayed; negative = audio should be advanced.
Offset is computed in video frames at a forced 25fps (the pipeline
re-encodes every clip to 25fps internally), then converted to ms.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

FPS = 25.0
OFFSET_RE = re.compile(r"AV offset:\s*(-?\d+)")
CONF_RE = re.compile(r"Confidence:\s*([\d.]+)")


def find_syncnet_dir(explicit):
    if explicit:
        return explicit
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "syncnet_python")


def check_install(syncnet_dir):
    model = os.path.join(syncnet_dir, "data", "syncnet_v2.model")
    face_weights = os.path.join(syncnet_dir, "detectors", "s3fd", "weights", "sfd_face.pth")
    missing = [p for p in (syncnet_dir, model, face_weights) if not os.path.exists(p)]
    if missing:
        print("error: SyncNet is not installed. Missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("Run: vlcsync.sh install-syncnet", file=sys.stderr)
        sys.exit(1)


def trim_clip(video, start, duration, out_path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start), "-t", str(duration), "-i", video,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {proc.stderr}")


def run_syncnet_pipeline(syncnet_dir, python_bin, clip_path, data_dir, reference, vshift,
                          facedet_scale, min_track):
    pipeline_cmd = [
        python_bin, "run_pipeline.py",
        "--videofile", clip_path, "--reference", reference, "--data_dir", data_dir,
        "--facedet_scale", str(facedet_scale), "--min_track", str(min_track),
    ]
    p1 = subprocess.run(pipeline_cmd, cwd=syncnet_dir, capture_output=True, text=True)
    log = p1.stdout + p1.stderr
    if p1.returncode != 0:
        print(log, file=sys.stderr)
        raise RuntimeError(
            "run_pipeline.py failed -- see log above. This usually means no face was "
            "detected in this window; try a different --start/--duration."
        )

    syncnet_cmd = [
        python_bin, "run_syncnet.py",
        "--videofile", clip_path, "--reference", reference, "--data_dir", data_dir,
        "--vshift", str(vshift),
    ]
    p2 = subprocess.run(syncnet_cmd, cwd=syncnet_dir, capture_output=True, text=True)
    log += "\n" + p2.stdout + p2.stderr
    if p2.returncode != 0:
        print(log, file=sys.stderr)
        raise RuntimeError("run_syncnet.py failed -- see log above.")
    return log


def parse_tracks(log):
    offsets = [int(x) for x in OFFSET_RE.findall(log)]
    confs = [float(x) for x in CONF_RE.findall(log)]
    return list(zip(offsets, confs))


def main():
    ap = argparse.ArgumentParser(description="SyncNet-based A/V offset detection (open-source model).")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0, help="Analysis window start, seconds (default: 0)")
    ap.add_argument("--duration", type=float, default=20.0,
                     help="Analysis window length, seconds (default: 20; needs ~4s+ of continuous "
                          "face detection to form a usable track)")
    ap.add_argument("--max-offset", type=float, default=2000.0, help="Max offset to search, ms (default: 2000)")
    ap.add_argument("--facedet-scale", type=float, default=0.25,
                     help="Downscale factor for face detection input (default: 0.25, upstream default). "
                          "Lower = faster face detection, may miss smaller/farther faces.")
    ap.add_argument("--min-track", type=int, default=100,
                     help="Minimum continuous face-track length in frames (default: 100 = 4s at 25fps, "
                          "upstream default). Lower allows shorter windows to produce a usable track, "
                          "at the cost of tracking segments with less speech to correlate against.")
    ap.add_argument("--syncnet-dir", default=None, help="Path to the cloned syncnet_python repo")
    ap.add_argument("--python-bin", default=sys.executable, help="Python interpreter to run the SyncNet scripts with")
    ap.add_argument("--keep-temp", action="store_true", help="Keep the temp working directory (for debugging)")
    ap.add_argument("--json", action="store_true", help="Print full JSON report to stdout instead of just the offset")
    args = ap.parse_args()

    syncnet_dir = find_syncnet_dir(args.syncnet_dir)
    check_install(syncnet_dir)

    vshift = max(1, round(args.max_offset / 1000.0 * FPS))
    workdir = tempfile.mkdtemp(prefix="vlcsync_syncnet_")
    reference = "vlcsync_" + str(int(time.time()))
    clip_path = os.path.join(workdir, "clip.mp4")

    try:
        print(f"trimming clip [{args.start:.1f}s .. {args.start + args.duration:.1f}s]...", file=sys.stderr)
        trim_clip(args.video, args.start, args.duration, clip_path)

        print("running SyncNet pipeline (face detection + tracking + sync scoring)...", file=sys.stderr)
        data_dir = os.path.join(workdir, "data")
        log = run_syncnet_pipeline(syncnet_dir, args.python_bin, clip_path, data_dir, reference, vshift,
                                    args.facedet_scale, args.min_track)

        tracks = parse_tracks(log)
        if not tracks:
            print(log, file=sys.stderr)
            print(
                "error: no face track produced a sync score. Try a window with a longer, closer, "
                "unobstructed view of the talking face (needs ~4s+ of continuous face detection).",
                file=sys.stderr,
            )
            sys.exit(1)

        best_offset_frames, best_conf = max(tracks, key=lambda t: t[1])
        offset_ms = best_offset_frames * (1000.0 / FPS)

        confidence = "low"
        if best_conf >= 5:
            confidence = "high"
        elif best_conf >= 2:
            confidence = "medium"

        report = {
            "offset_ms": round(offset_ms, 1),
            "syncnet_confidence": round(best_conf, 3),
            "confidence": confidence,
            "tracks_found": len(tracks),
            "window_start_s": args.start,
            "window_duration_s": args.duration,
            "model": "syncnet",
            "note": "positive offset_ms = delay audio; negative = advance audio",
        }
        print(json.dumps(report, indent=2), file=sys.stderr)

        if len(tracks) > 1:
            print(
                f"warning: {len(tracks)} face tracks found in this window; picked the "
                "highest-confidence one. If that's the wrong person, narrow --start/--duration "
                "to a segment with only the intended speaker on screen.",
                file=sys.stderr,
            )
        if confidence == "low":
            print("warning: low SyncNet confidence -- try a clearer/closer window of the talking face",
                  file=sys.stderr)

        if args.json:
            print(json.dumps(report))
        else:
            print(report["offset_ms"])
    finally:
        if not args.keep_temp:
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"kept temp dir: {workdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
