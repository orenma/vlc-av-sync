#!/usr/bin/env python3
"""
Finds a consensus A/V offset by combining two ideas that measured out much
faster than one long window or blind evenly-spaced windows:

1. Cheap probing first. The expensive part of --model syncnet is its own
   face detector (S3FD via PyTorch), which runs at only ~11fps on CPU --
   a full 20s window costs ~45s just for that step, and a heavily-edited
   file can burn through several failed windows before finding a usable
   one. MediaPipe (already used by the heuristic backend) does the same
   "is there a continuously-visible face, and is there speech" check in
   ~5s for an 8s clip. So: probe many short, cheap candidate windows across
   the file with the heuristic backend, keep only its face-coverage/
   voiced-fraction numbers (ignore its offset answer, which isn't reliable
   enough to trust -- see README), and only run the real, expensive
   detector on the candidates that already look good.

2. Parallelism. Both the probe pass and the real-detection pass are
   independent per-window CPU-bound subprocesses, so they're run
   concurrently (bounded by CPU count), with each subprocess's internal
   thread pool capped so they don't oversubscribe the machine fighting
   each other.

Consensus is the median of the successful real-detection runs; cross-window
agreement is reported because it proved to be a better confidence signal in
testing than either backend's own per-run confidence score.
"""

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEURISTIC_PY = os.path.join(SCRIPT_DIR, "av_sync_detect.py")
SYNCNET_PY = os.path.join(SCRIPT_DIR, "syncnet_detect.py")

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(video_path):
    proc = subprocess.run(["ffmpeg", "-i", video_path], capture_output=True, text=True)
    m = DURATION_RE.search(proc.stderr)
    if not m:
        raise RuntimeError(f"could not determine duration of {video_path} via ffmpeg -i")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def thread_capped_env(thread_cap):
    """Caps each subprocess's internal thread pool (torch/OpenBLAS/etc.) so
    running several of these concurrently doesn't oversubscribe the CPU."""
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(thread_cap)
    return env


def run_json_subprocess(cmd, env):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    if proc.returncode != 0:
        return None
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def probe_one(python_bin, video, start, duration, sample_fps, env):
    cmd = [python_bin, HEURISTIC_PY, video, "--start", str(start), "--duration", str(duration),
           "--sample-fps", str(sample_fps), "--json"]
    report = run_json_subprocess(cmd, env)
    if report is None:
        return {"start_s": start, "face_coverage": 0.0, "voiced_fraction": 0.0}
    return {
        "start_s": start,
        "face_coverage": report.get("face_coverage", 0.0),
        "voiced_fraction": report.get("voiced_fraction", 0.0),
    }


def detect_one(python_bin, script, video, start, duration, max_offset, extra_args, env):
    cmd = [python_bin, script, video, "--start", str(start), "--duration", str(duration),
           "--max-offset", str(max_offset), "--json"] + extra_args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True, env=env)
    if proc.returncode != 0:
        return {"start_s": start, "ok": False}
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return {"start_s": start, "ok": False}
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"start_s": start, "ok": False}
    report["start_s"] = start
    report["ok"] = True
    return report


def run_parallel(fn_calls, max_workers):
    """fn_calls: list of zero-arg callables. Returns results in the same order."""
    results = [None] * len(fn_calls)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn): i for i, fn in enumerate(fn_calls)}
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    return results


def select_candidates(probes, num_windows, min_spacing, min_face_coverage, min_voiced_fraction):
    scored = sorted(probes, key=lambda p: min(p["face_coverage"], p["voiced_fraction"]), reverse=True)
    passing = [p for p in scored
               if p["face_coverage"] >= min_face_coverage and p["voiced_fraction"] >= min_voiced_fraction]
    pool = passing if len(passing) >= num_windows else scored  # fall back to best-available if too few pass

    selected = []
    for p in pool:
        if all(abs(p["start_s"] - s["start_s"]) >= min_spacing for s in selected):
            selected.append(p)
        if len(selected) >= num_windows:
            break
    return selected, len(passing)


def main():
    ap = argparse.ArgumentParser(description="Probe for good windows, then find a consensus A/V offset fast.")
    ap.add_argument("video")
    ap.add_argument("--model", choices=["heuristic", "syncnet"], default="syncnet",
                     help="backend for the real (post-probe) detection (default: syncnet)")
    ap.add_argument("--windows", type=int, default=6, help="number of good windows to detect on (default: 6)")
    ap.add_argument("--duration", type=float, default=10.0,
                     help="each real-detection window's length, seconds (default: 10 -- short since "
                          "probing already confirmed good coverage there)")
    ap.add_argument("--max-offset", type=float, default=6000.0,
                     help="max offset to search per window, ms (default: 6000)")
    ap.add_argument("--margin", type=float, default=60.0,
                     help="skip this many seconds at the start/end of the file (default: 60)")
    ap.add_argument("--probe-duration", type=float, default=8.0, help="probe clip length, seconds (default: 8)")
    ap.add_argument("--probe-fps", type=float, default=6.0, help="probe sampling rate (default: 6)")
    ap.add_argument("--probe-spacing", type=float, default=45.0,
                     help="seconds between candidate probes across the file (default: 45)")
    ap.add_argument("--max-candidates", type=int, default=80,
                     help="cap on probes run regardless of file length (default: 80)")
    ap.add_argument("--min-face-coverage", type=float, default=0.7,
                     help="probe threshold to shortlist a window (default: 0.7)")
    ap.add_argument("--min-voiced-fraction", type=float, default=0.3,
                     help="probe threshold to shortlist a window (default: 0.3)")
    ap.add_argument("--jobs", type=int, default=min(6, max(1, (os.cpu_count() or 4) - 1)),
                     help="max concurrent subprocesses (default: cpu_count-1, capped at 6)")
    ap.add_argument("--agree-tolerance", type=float, default=300.0,
                     help="ms tolerance for counting a window as agreeing with the median (default: 300)")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--json", action="store_true", help="print full JSON summary to stdout instead of just the offset")
    args, extra = ap.parse_known_args()

    script = SYNCNET_PY if args.model == "syncnet" else HEURISTIC_PY
    env = thread_capped_env(max(1, (os.cpu_count() or 4) // args.jobs))

    total_duration = probe_duration(args.video)
    usable_end = max(args.margin, total_duration - args.margin - args.probe_duration)
    n_candidates = min(args.max_candidates, max(1, int((usable_end - args.margin) / args.probe_spacing) + 1))
    candidate_starts = [args.margin + i * (usable_end - args.margin) / max(1, n_candidates - 1)
                         for i in range(n_candidates)] if n_candidates > 1 else [args.margin]

    print(f"probing {len(candidate_starts)} candidate window(s) ({args.probe_duration:.0f}s each) "
          f"across {total_duration:.0f}s of video, {args.jobs} at a time...", file=sys.stderr)
    probes = run_parallel(
        [lambda s=s: probe_one(args.python_bin, args.video, s, args.probe_duration, args.probe_fps, env)
         for s in candidate_starts],
        args.jobs,
    )

    # Require real spread across the file, not just enough gap to avoid literal
    # overlap -- otherwise several "independent" windows can all land in the same
    # scene and their agreement isn't much of a cross-check.
    min_spacing = max(args.duration * 2, total_duration / (args.windows * 3))
    # The cheap probe (short clip, MediaPipe) and the real detector (longer
    # clip, and for syncnet a strict 4s *continuous* track requirement) don't
    # perfectly agree -- a window can pass the probe and still fail real
    # detection. Oversample so a few such misses don't starve the consensus.
    selected, n_passing = select_candidates(
        probes, args.windows * 2, min_spacing=min_spacing,
        min_face_coverage=args.min_face_coverage, min_voiced_fraction=args.min_voiced_fraction,
    )
    print(f"{n_passing}/{len(probes)} candidates passed coverage thresholds; "
          f"selected {len(selected)} spread-out window(s) for real detection "
          f"(oversampled from a target of {args.windows})", file=sys.stderr)
    if n_passing < args.windows:
        print(f"warning: fewer than --windows={args.windows} candidates cleared the coverage thresholds -- "
              "using the best available anyway; results may be less reliable", file=sys.stderr)

    print(f"running --model {args.model} on {len(selected)} window(s), {args.jobs} at a time...", file=sys.stderr)
    results = run_parallel(
        [lambda s=c["start_s"]: detect_one(args.python_bin, script, args.video, s, args.duration,
                                            args.max_offset, extra, env)
         for c in selected],
        args.jobs,
    )

    successes = [r for r in results if r.get("ok")]
    offsets = [r["offset_ms"] for r in successes]

    summary = {
        "model": args.model,
        "candidates_probed": len(probes),
        "candidates_passing": n_passing,
        "windows_run": len(results),
        "windows_succeeded": len(successes),
        "per_window": [
            {"start_s": round(r.get("start_s", 0), 1), "ok": r.get("ok", False),
             "offset_ms": r.get("offset_ms"), "confidence": r.get("confidence")}
            for r in results
        ],
    }

    if not offsets:
        summary["consensus_offset_ms"] = None
        print(json.dumps(summary, indent=2), file=sys.stderr)
        print("error: no window produced a result -- try more --windows, a lower --min-face-coverage, "
              "or a different --model", file=sys.stderr)
        sys.exit(1)

    median_offset = statistics.median(offsets)
    agreeing = [o for o in offsets if abs(o - median_offset) <= args.agree_tolerance]

    summary["consensus_offset_ms"] = round(median_offset, 1)
    summary["agreement"] = f"{len(agreeing)}/{len(offsets)}"
    summary["offsets_ms"] = offsets
    summary["spread_ms"] = round(max(offsets) - min(offsets), 1) if len(offsets) > 1 else 0.0

    print(json.dumps(summary, indent=2), file=sys.stderr)

    if len(agreeing) < len(offsets) / 2.0:
        print(f"warning: only {len(agreeing)}/{len(offsets)} windows agree within "
              f"{args.agree_tolerance:.0f}ms of the median -- results are inconsistent, treat the "
              "consensus with caution (try more --windows, or --model syncnet if you used heuristic)",
              file=sys.stderr)

    if args.json:
        print(json.dumps(summary))
    else:
        print(summary["consensus_offset_ms"])


if __name__ == "__main__":
    main()
