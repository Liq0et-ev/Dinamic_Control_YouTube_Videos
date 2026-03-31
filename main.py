#!/usr/bin/env python3
"""
==================================================================
  Dynamic Highlight Extraction System — Batch Edition
  ====================================================
  Scans a SOURCE_DIR for video files, lets the user choose which
  ones to process (all / single / custom list), analyses multimodal
  signals, and exports ~1-minute highlight clips to TARGET_DIR.

  Usage:
      python main.py              (interactive case selection)
      python main.py --case all   (process every video)
      python main.py --case 1     (process video #1 from the list)
      python main.py --case 1,3,5 (process videos #1, #3, #5)

  Configuration:
      Edit SOURCE_DIR and TARGET_DIR below to point at your folders.
==================================================================

Architecture
------------
1. Directory scan    → find all video files in SOURCE_DIR
2. Case selection    → user picks which files to process
3. Per-video pipeline:
     a) Extract visual features (motion intensity + optical flow)
     b) Extract audio features  (RMS energy + spectral flux)
     c) DSP: normalise → SG-smooth → derivatives → adaptive threshold
     d) Detect dynamic segments (Weighted Heuristic by default)
     e) Temporal: median filter → +5s buffer → merge → ~60s target
     f) Export individual clips to TARGET_DIR
     g) Save diagnostic plot
4. Processing log    → persistent JSON tracking completed files
"""

import gc
import os
import sys
import time

import numpy as np

from video_features import extract_video_features
from audio_features import extract_audio_features
from dsp_pipeline import (
    resample_to_common_axis,
    minmax_norm,
    smooth_sg,
    derivatives,
    adaptive_threshold_fast,
)
from detectors import (
    detect_weighted_heuristic,
    detect_surprisal,
    detect_multimodal_attention,
)
from temporal import build_extraction_map
from diagnostics import plot_diagnostics
from processing_log import (
    is_completed,
    mark_completed,
    mark_failed,
    print_log_summary,
)


# ================================================================== #
#  CONFIGURATION — edit these two paths to match your setup
# ================================================================== #
SOURCE_DIR = r"./source_videos"       # ← folder with your raw videos
TARGET_DIR = r"./exported_highlights"  # ← folder for output clips
# ================================================================== #

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}

# Default processing parameters
DEFAULT_VARIANT       = "heuristic"   # "heuristic" | "surprisal" | "attention"
DEFAULT_TARGET_SEC    = 60.0
DEFAULT_BUFFER_SEC    = 5.0
DEFAULT_MERGE_GAP_SEC = 3.0
DEFAULT_THRESHOLD_K   = 1.5
DEFAULT_SAMPLE_FPS    = 2.0
DEFAULT_SG_WINDOW     = 15


# ================================================================== #
#  Directory Management
# ================================================================== #
def discover_videos(source_dir: str) -> list[str]:
    """
    Traverse SOURCE_DIR and return sorted list of video file paths.
    Only files with recognised video extensions are included.
    """
    if not os.path.isdir(source_dir):
        print(f"\n  ERROR: Source directory not found: {source_dir}")
        print(f"  Create it and place your video files inside, then re-run.\n")
        sys.exit(1)

    videos = []
    for entry in sorted(os.listdir(source_dir)):
        full_path = os.path.join(source_dir, entry)
        if os.path.isfile(full_path):
            _, ext = os.path.splitext(entry)
            if ext.lower() in VIDEO_EXTENSIONS:
                videos.append(full_path)

    return videos


def ensure_target_dir(target_dir: str):
    """Create target directory (and parents) if it doesn't exist."""
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"  Created output directory: {target_dir}")


# ================================================================== #
#  File Listing & Case Selection
# ================================================================== #
def print_file_list(videos: list[str], target_dir: str):
    """Display a numbered list of discovered videos."""
    print("\n" + "=" * 64)
    print("  DYNAMIC HIGHLIGHT EXTRACTION — Batch Processor")
    print("=" * 64)
    print(f"\n  Source : {os.path.abspath(SOURCE_DIR)}")
    print(f"  Target : {os.path.abspath(TARGET_DIR)}")
    print(f"  Videos found: {len(videos)}\n")

    if not videos:
        print("  No video files found in SOURCE_DIR.")
        print(f"  Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}\n")
        sys.exit(0)

    print("  ┌─────┬────────────────────────────────────────┬──────────┐")
    print("  │  #  │  Filename                              │  Status  │")
    print("  ├─────┼────────────────────────────────────────┼──────────┤")

    for i, path in enumerate(videos, 1):
        name = os.path.basename(path)
        # Truncate long names
        display_name = name if len(name) <= 38 else name[:35] + "..."
        status = " DONE" if is_completed(target_dir, name) else "  new"
        print(f"  │ {i:>3} │  {display_name:<38} │ {status:>6}  │")

    print("  └─────┴────────────────────────────────────────┴──────────┘")

    # Show processing log summary
    print("\n  Processing history:")
    print_log_summary(target_dir)
    print()


def parse_case_selection(case_input: str, total: int) -> list[int]:
    """
    Parse user's case selection into a list of 0-based indices.

    Supported formats:
      "all"       → all videos
      "3"         → single video #3
      "1,3,5"     → videos #1, #3, #5
      "2-5"       → videos #2 through #5
      "1,3-5,8"   → mixed: #1, #3, #4, #5, #8
    """
    case_input = case_input.strip().lower()

    if case_input in ("all", "a", "*"):
        return list(range(total))

    indices = set()
    for part in case_input.split(","):
        part = part.strip()
        if "-" in part:
            # Range: "2-5"
            try:
                a, b = part.split("-", 1)
                a, b = int(a.strip()), int(b.strip())
                for n in range(a, b + 1):
                    if 1 <= n <= total:
                        indices.add(n - 1)
            except ValueError:
                print(f"  Warning: could not parse range '{part}', skipping.")
        else:
            # Single number
            try:
                n = int(part)
                if 1 <= n <= total:
                    indices.add(n - 1)
                else:
                    print(f"  Warning: #{n} is out of range (1-{total}), skipping.")
            except ValueError:
                # Try matching by filename substring
                pass

    return sorted(indices)


def get_selection_interactive(videos: list[str]) -> list[int]:
    """Prompt user to choose which videos to process."""
    print("  Select videos to process:")
    print("    'all'       → process all videos")
    print("    '3'         → process video #3 only")
    print("    '1,3,5'     → process videos #1, #3, and #5")
    print("    '2-5'       → process videos #2 through #5")
    print("    '1,3-5,8'   → mixed selection")
    print("    'q'         → quit\n")

    while True:
        try:
            choice = input("  Your selection: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            sys.exit(0)

        if choice.lower() in ("q", "quit", "exit"):
            print("  Exiting.")
            sys.exit(0)

        indices = parse_case_selection(choice, len(videos))
        if indices:
            names = [os.path.basename(videos[i]) for i in indices]
            print(f"\n  Selected {len(indices)} video(s):")
            for idx in indices:
                print(f"    → {os.path.basename(videos[idx])}")
            print()
            return indices
        else:
            print("  No valid selection. Try again.\n")


# ================================================================== #
#  Clip Export — Individual Segments
# ================================================================== #
def export_individual_clips(video_path: str, intervals: list,
                            target_dir: str, base_name: str) -> int:
    """
    Export each interval as a separate clip file:
      base_name_highlight_1.mp4
      base_name_highlight_2.mp4
      ...

    Uses ffmpeg stream-copy (no re-encode) for speed.
    Falls back to MoviePy if ffmpeg is not available.

    Returns number of clips exported.
    """
    if not intervals:
        print("[export] No dynamic segments detected — nothing to export.")
        return 0

    exported = 0

    # Try ffmpeg first (fast, no re-encode)
    try:
        import subprocess

        for i, (s, e) in enumerate(intervals, 1):
            clip_name = f"{base_name}_highlight_{i}.mp4"
            clip_path = os.path.join(target_dir, clip_name)

            duration = e - s
            if duration < 0.5:
                print(f"  Segment {i}: {s:.1f}s→{e:.1f}s — too short, skipping")
                continue

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{s:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                clip_path,
            ]
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            if result.returncode == 0:
                print(f"  Segment {i}: {s:.1f}s → {e:.1f}s "
                      f"({duration:.1f}s) → {clip_name}")
                exported += 1
            else:
                # Fallback: re-encode this segment
                cmd_reencode = [
                    "ffmpeg", "-y",
                    "-ss", f"{s:.3f}",
                    "-i", video_path,
                    "-t", f"{duration:.3f}",
                    "-c:v", "libx264", "-preset", "fast",
                    "-c:a", "aac",
                    clip_path,
                ]
                result2 = subprocess.run(
                    cmd_reencode, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if result2.returncode == 0:
                    print(f"  Segment {i}: {s:.1f}s → {e:.1f}s "
                          f"({duration:.1f}s) → {clip_name} (re-encoded)")
                    exported += 1
                else:
                    print(f"  Segment {i}: FAILED to export")

        return exported

    except FileNotFoundError:
        print("[export] ffmpeg not found, falling back to MoviePy...")
        return _export_clips_moviepy(video_path, intervals, target_dir, base_name)


def _export_clips_moviepy(video_path: str, intervals: list,
                          target_dir: str, base_name: str) -> int:
    """Fallback: export clips using MoviePy (requires re-encoding)."""
    from moviepy.editor import VideoFileClip

    source = VideoFileClip(video_path)
    exported = 0

    for i, (s, e) in enumerate(intervals, 1):
        clip_name = f"{base_name}_highlight_{i}.mp4"
        clip_path = os.path.join(target_dir, clip_name)

        s = max(0, s)
        e = min(e, source.duration)
        if e - s < 0.5:
            continue

        clip = source.subclip(s, e)
        clip.write_videofile(clip_path, codec="libx264",
                             audio_codec="aac", logger=None)
        clip.close()
        print(f"  Segment {i}: {s:.1f}s → {e:.1f}s → {clip_name}")
        exported += 1

    source.close()
    return exported


# ================================================================== #
#  Single-Video Processing Pipeline
# ================================================================== #
def process_single_video(
    video_path: str,
    target_dir: str,
    variant: str = DEFAULT_VARIANT,
    target_sec: float = DEFAULT_TARGET_SEC,
    buffer_sec: float = DEFAULT_BUFFER_SEC,
    merge_gap_sec: float = DEFAULT_MERGE_GAP_SEC,
    threshold_k: float = DEFAULT_THRESHOLD_K,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    sg_window: int = DEFAULT_SG_WINDOW,
) -> tuple[int, float]:
    """
    Full analysis pipeline for one video file.

    Returns (clips_exported, total_clip_duration).
    """
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]

    t0 = time.time()

    # ── STEP 1: Feature Extraction ──────────────────────────────── #
    print(f"\n  {'─' * 52}")
    print(f"  STEP 1/5 — Feature Extraction")
    print(f"  {'─' * 52}")

    vid_ts, motion_raw, flow_raw, native_fps, total_dur = \
        extract_video_features(video_path, sample_fps=sample_fps)

    aud_ts, rms_raw, flux_raw = extract_audio_features(video_path)

    # ── STEP 2: Signal Processing (DSP) ─────────────────────────── #
    print(f"\n  {'─' * 52}")
    print(f"  STEP 2/5 — Signal Processing (DSP)")
    print(f"  {'─' * 52}")

    dt = 1.0 / sample_fps
    common_ts = np.arange(0, total_dur, dt)

    # Resample to common axis
    motion = resample_to_common_axis(vid_ts, motion_raw, common_ts)
    flow   = resample_to_common_axis(vid_ts, flow_raw, common_ts)
    rms    = resample_to_common_axis(aud_ts, rms_raw, common_ts)
    flux   = resample_to_common_axis(aud_ts, flux_raw, common_ts)

    # Normalise (Min-Max → [0,1])
    motion = minmax_norm(motion)
    flow   = minmax_norm(flow)
    rms    = minmax_norm(rms)
    flux   = minmax_norm(flux)

    # Savitzky-Golay smoothing
    motion = smooth_sg(motion, window=sg_window)
    flow   = smooth_sg(flow, window=sg_window)
    rms    = smooth_sg(rms, window=sg_window)
    flux   = smooth_sg(flux, window=sg_window)

    # Derivatives (onset detection support)
    motion_d1, motion_d2 = derivatives(motion, dt)
    flow_d1, flow_d2     = derivatives(flow, dt)

    print(f"  [dsp] {len(common_ts)} samples, dt={dt:.2f}s, "
          f"duration={total_dur:.1f}s")

    # ── STEP 3: Dynamic Segment Detection ────────────────────────── #
    print(f"\n  {'─' * 52}")
    print(f"  STEP 3/5 — Detection ({variant})")
    print(f"  {'─' * 52}")

    variant_map = {
        "heuristic": ("Weighted Heuristic (A)", detect_weighted_heuristic),
        "surprisal": ("Surprisal (B)", detect_surprisal),
        "attention": ("Multimodal Attention (C)", detect_multimodal_attention),
    }

    variant_name, detect_fn = variant_map[variant]

    mask = detect_fn(
        motion, flow, rms, flux,
        dt=dt, threshold_k=threshold_k,
    )

    active_pct = mask.sum() / len(mask) * 100
    print(f"  [detect] {active_pct:.1f}% of video flagged as dynamic")

    # ── STEP 4: Temporal Post-Processing ─────────────────────────── #
    print(f"\n  {'─' * 52}")
    print(f"  STEP 4/5 — Temporal Post-Processing")
    print(f"  {'─' * 52}")

    intervals, clip_dur, raw_intervals = build_extraction_map(
        mask, common_ts, total_dur,
        buffer_sec=buffer_sec,
        merge_gap_sec=merge_gap_sec,
        target_sec=target_sec,
    )

    print(f"  [temporal] {len(raw_intervals)} raw → {len(intervals)} final segments")
    print(f"  [temporal] Total clip duration: {clip_dur:.1f}s (target: {target_sec:.0f}s)")

    # ── STEP 5: Export ───────────────────────────────────────────── #
    print(f"\n  {'─' * 52}")
    print(f"  STEP 5/5 — Exporting Clips")
    print(f"  {'─' * 52}")

    # Create per-video subfolder in target dir
    video_output_dir = os.path.join(target_dir, base_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Export individual highlight clips
    clips_exported = export_individual_clips(
        video_path, intervals, video_output_dir, base_name
    )

    # Diagnostic plot
    composite_w = {"motion": 0.30, "flow": 0.30, "rms": 0.20, "flux": 0.20}
    composite = (composite_w["motion"] * motion + composite_w["flow"] * flow
                 + composite_w["rms"] * rms + composite_w["flux"] * flux)
    threshold = adaptive_threshold_fast(composite, window_sec=30.0,
                                        dt=dt, k=threshold_k)

    plot_path = os.path.join(video_output_dir, f"{base_name}_diagnostics.png")
    plot_diagnostics(
        timestamps=common_ts,
        motion=motion, flow=flow, rms=rms, flux=flux,
        composite=composite, threshold=threshold,
        mask=mask, intervals=intervals, raw_intervals=raw_intervals,
        video_duration=total_dur,
        output_path=plot_path,
        variant_name=variant_name,
    )

    elapsed = time.time() - t0
    print(f"\n  ✓ {filename} processed in {elapsed:.1f}s")
    print(f"    Clips: {clips_exported} | Duration: {clip_dur:.1f}s | "
          f"Output: {video_output_dir}")

    return clips_exported, clip_dur


# ================================================================== #
#  Batch Orchestrator
# ================================================================== #
def run_batch(
    videos: list[str],
    selected_indices: list[int],
    target_dir: str,
    skip_completed: bool = True,
    **pipeline_kwargs,
):
    """
    Process multiple videos in sequence.
    Properly releases memory between files.
    """
    total = len(selected_indices)
    results = []

    print("\n" + "=" * 64)
    print(f"  BATCH PROCESSING — {total} video(s)")
    print("=" * 64)

    for job_num, idx in enumerate(selected_indices, 1):
        video_path = videos[idx]
        filename = os.path.basename(video_path)

        print(f"\n{'▓' * 64}")
        print(f"  VIDEO {job_num}/{total}: {filename}")
        print(f"{'▓' * 64}")

        # Skip already-processed files
        if skip_completed and is_completed(target_dir, filename):
            print(f"  ⏭  Already processed — skipping. (Use --force to re-process)")
            results.append((filename, "skipped", 0, 0.0))
            continue

        try:
            clips, duration = process_single_video(
                video_path, target_dir, **pipeline_kwargs
            )
            mark_completed(target_dir, filename, video_path,
                           clips, duration)
            results.append((filename, "completed", clips, duration))

        except Exception as e:
            print(f"\n  ✗ ERROR processing {filename}: {e}")
            mark_failed(target_dir, filename, video_path, str(e))
            results.append((filename, "failed", 0, 0.0))

        finally:
            # Release memory between files
            gc.collect()

    # ── Summary ──────────────────────────────────────────────────── #
    print("\n" + "=" * 64)
    print("  BATCH SUMMARY")
    print("=" * 64)
    print(f"\n  {'File':<40} {'Status':<12} {'Clips':<7} {'Duration'}")
    print(f"  {'─'*40} {'─'*12} {'─'*7} {'─'*10}")

    total_clips = 0
    total_duration = 0.0
    for name, status, clips, dur in results:
        display = name if len(name) <= 38 else name[:35] + "..."
        print(f"  {display:<40} {status:<12} {clips:<7} {dur:.1f}s")
        if status == "completed":
            total_clips += clips
            total_duration += dur

    completed = sum(1 for _, s, _, _ in results if s == "completed")
    skipped = sum(1 for _, s, _, _ in results if s == "skipped")
    failed = sum(1 for _, s, _, _ in results if s == "failed")

    print(f"\n  Completed: {completed} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Total clips exported: {total_clips}")
    print(f"  Total highlight duration: {total_duration:.1f}s")
    print(f"  Output directory: {os.path.abspath(target_dir)}")
    print("=" * 64 + "\n")


# ================================================================== #
#  CLI Entry Point
# ================================================================== #
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Dynamic Highlight Extraction — extract ~1-min "
                    "highlights from all videos in a source folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Case Selection:
  --case all         Process every video in SOURCE_DIR
  --case 3           Process only video #3
  --case 1,3,5       Process videos #1, #3, and #5
  --case 2-5         Process videos #2 through #5
  --case 1,3-5,8     Mixed selection
  (omit --case)      Interactive mode — shows list, prompts for choice

Detection Variants:
  heuristic  (A)     Weighted Heuristic — best for diverse content (DEFAULT)
  surprisal  (B)     Information-theoretic — good for uniform backgrounds
  attention  (C)     Gradient-based attention — good for multi-source

Examples:
  python main.py                           # interactive selection
  python main.py --case all                # process everything
  python main.py --case 1,3 --variant attention --target 90
  python main.py --case all --force        # re-process even if done
        """,
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Video selection: 'all', single number, comma-separated list, "
             "or range (e.g. '2-5'). Omit for interactive mode.",
    )
    parser.add_argument(
        "--variant", choices=["heuristic", "surprisal", "attention"],
        default=DEFAULT_VARIANT,
        help=f"Detection algorithm (default: {DEFAULT_VARIANT})",
    )
    parser.add_argument(
        "--target", type=float, default=DEFAULT_TARGET_SEC,
        help=f"Target clip length in seconds (default: {DEFAULT_TARGET_SEC})",
    )
    parser.add_argument(
        "--buffer", type=float, default=DEFAULT_BUFFER_SEC,
        help=f"Post-event safety buffer (default: {DEFAULT_BUFFER_SEC}s)",
    )
    parser.add_argument(
        "--merge-gap", type=float, default=DEFAULT_MERGE_GAP_SEC,
        help=f"Max gap to merge segments (default: {DEFAULT_MERGE_GAP_SEC}s)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD_K,
        help=f"Sensitivity k (default: {DEFAULT_THRESHOLD_K}). "
             "Lower = more clips; higher = only strongest peaks.",
    )
    parser.add_argument(
        "--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS,
        help=f"Video analysis sample rate (default: {DEFAULT_SAMPLE_FPS} fps)",
    )
    parser.add_argument(
        "--sg-window", type=int, default=DEFAULT_SG_WINDOW,
        help=f"Savitzky-Golay window (default: {DEFAULT_SG_WINDOW}, must be odd)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process videos even if already completed",
    )
    parser.add_argument(
        "--source-dir", type=str, default=None,
        help=f"Override SOURCE_DIR (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--target-dir", type=str, default=None,
        help=f"Override TARGET_DIR (default: {TARGET_DIR})",
    )

    args = parser.parse_args()

    # Resolve directories
    source = args.source_dir or SOURCE_DIR
    target = args.target_dir or TARGET_DIR

    # Discover videos
    videos = discover_videos(source)
    ensure_target_dir(target)

    # Display file list
    print_file_list(videos, target)

    if not videos:
        return

    # Case selection
    if args.case is not None:
        selected = parse_case_selection(args.case, len(videos))
        if not selected:
            print("  No valid videos matched your selection.")
            sys.exit(1)
        print(f"  Selected {len(selected)} video(s) via --case '{args.case}':")
        for idx in selected:
            print(f"    → {os.path.basename(videos[idx])}")
        print()
    else:
        selected = get_selection_interactive(videos)

    # Build pipeline kwargs
    pipeline_kwargs = dict(
        variant=args.variant,
        target_sec=args.target,
        buffer_sec=args.buffer,
        merge_gap_sec=args.merge_gap,
        threshold_k=args.threshold,
        sample_fps=args.sample_fps,
        sg_window=args.sg_window,
    )

    # Run batch
    run_batch(
        videos=videos,
        selected_indices=selected,
        target_dir=target,
        skip_completed=not args.force,
        **pipeline_kwargs,
    )


if __name__ == "__main__":
    main()
