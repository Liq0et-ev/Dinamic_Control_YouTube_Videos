#!/usr/bin/env python3
"""
=============================================================
  Dynamic Highlight Extraction System
  ====================================
  Automatically extracts ~1-minute highlight clips from
  long-form video by analysing visual AND audio dynamics.

  Usage:
      python extract_highlights.py input_video.mp4 [options]

  Run with --help for full option list.
=============================================================

Architecture Overview
---------------------
1. **Perception** – Extract 4 feature channels:
     Visual:  motion intensity (frame diff), optical flow (Farneback)
     Audio:   RMS energy, spectral flux

2. **Signal Processing (DSP)** – For each channel:
     a) Resample to a common 2-Hz time axis
     b) Min-Max normalise to [0,1]
     c) Savitzky-Golay smooth (preserves peak shape)
     d) Compute 1st/2nd derivatives for onset detection

3. **Detection** – Three selectable variants:
     (A) Weighted Heuristic   – default, best for unstructured video
     (B) Information-Theoretic Surprisal
     (C) Multimodal Attention

4. **Temporal Post-Processing**:
     - Median filter for segment coherence
     - +5 s safety buffer after each event
     - Interval merging for close segments
     - Target ~60 s total clip length

5. **Output**:
     - Concatenated highlight clip (MP4)
     - Diagnostic PNG with signal plots and extraction map
"""

import argparse
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


# ------------------------------------------------------------------ #
#  Clip export
# ------------------------------------------------------------------ #
def export_clip(video_path: str, intervals: list, output_path: str):
    """
    Concatenate selected intervals into a single MP4 using MoviePy.
    Falls back to ffmpeg CLI if MoviePy is unavailable.
    """
    if not intervals:
        print("[clip] No dynamic segments detected – nothing to export.")
        return

    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        print(f"[clip] Extracting {len(intervals)} segments with MoviePy...")
        source = VideoFileClip(video_path)
        clips = []
        for i, (s, e) in enumerate(intervals):
            s = max(0, s)
            e = min(e, source.duration)
            if e - s < 0.1:
                continue
            clips.append(source.subclip(s, e))
            print(f"  Segment {i+1}: {s:.1f}s → {e:.1f}s  ({e-s:.1f}s)")

        if not clips:
            print("[clip] All segments too short – nothing to export.")
            return

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        source.close()
        final.close()
        print(f"[clip] Highlight saved → {output_path}")

    except ImportError:
        print("[clip] MoviePy not available, falling back to ffmpeg concat...")
        _export_ffmpeg(video_path, intervals, output_path)


def _export_ffmpeg(video_path: str, intervals: list, output_path: str):
    """Fallback: use ffmpeg concat demuxer."""
    import subprocess
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="dyn_clip_")
    parts = []

    for i, (s, e) in enumerate(intervals):
        part_path = os.path.join(tmp_dir, f"part_{i:03d}.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(s), "-to", str(e),
            "-i", video_path,
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            part_path,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        parts.append(part_path)

    # Write concat list
    list_path = os.path.join(tmp_dir, "concat.txt")
    with open(list_path, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path, "-c", "copy", output_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Cleanup
    for p in parts:
        os.remove(p)
    os.remove(list_path)
    os.rmdir(tmp_dir)
    print(f"[clip] Highlight saved → {output_path}")


# ------------------------------------------------------------------ #
#  Main orchestrator
# ------------------------------------------------------------------ #
def run_pipeline(
    video_path: str,
    variant: str = "heuristic",
    target_sec: float = 60.0,
    buffer_sec: float = 5.0,
    merge_gap_sec: float = 3.0,
    threshold_k: float = 1.5,
    sample_fps: float = 2.0,
    sg_window: int = 15,
    output_dir: str | None = None,
):
    t0 = time.time()
    base = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(video_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================== #
    #  STEP 1 – Feature Extraction
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  STEP 1 / 5  –  Feature Extraction")
    print("=" * 60)

    vid_ts, motion_raw, flow_raw, native_fps, total_dur = \
        extract_video_features(video_path, sample_fps=sample_fps)

    aud_ts, rms_raw, flux_raw = extract_audio_features(video_path)

    # ============================================================== #
    #  STEP 2 – Signal Processing (DSP)
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  STEP 2 / 5  –  Signal Processing")
    print("=" * 60)

    # Common time axis at sample_fps
    dt = 1.0 / sample_fps
    common_ts = np.arange(0, total_dur, dt)

    # Resample all channels
    motion = resample_to_common_axis(vid_ts, motion_raw, common_ts)
    flow   = resample_to_common_axis(vid_ts, flow_raw, common_ts)
    rms    = resample_to_common_axis(aud_ts, rms_raw, common_ts)
    flux   = resample_to_common_axis(aud_ts, flux_raw, common_ts)

    # Normalise (Min-Max to [0,1])
    motion = minmax_norm(motion)
    flow   = minmax_norm(flow)
    rms    = minmax_norm(rms)
    flux   = minmax_norm(flux)

    # Smooth (Savitzky-Golay)
    motion = smooth_sg(motion, window=sg_window)
    flow   = smooth_sg(flow, window=sg_window)
    rms    = smooth_sg(rms, window=sg_window)
    flux   = smooth_sg(flux, window=sg_window)

    # Derivatives (for diagnostics / onset analysis)
    motion_d1, motion_d2 = derivatives(motion, dt)
    flow_d1, flow_d2     = derivatives(flow, dt)

    print(f"[dsp] {len(common_ts)} samples on common axis "
          f"(dt={dt:.2f}s, duration={total_dur:.1f}s)")

    # ============================================================== #
    #  STEP 3 – Dynamic Segment Detection
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  STEP 3 / 5  –  Detection")
    print("=" * 60)

    variant_map = {
        "heuristic": ("Weighted Heuristic (A)", detect_weighted_heuristic),
        "surprisal": ("Information-Theoretic Surprisal (B)", detect_surprisal),
        "attention": ("Multimodal Attention (C)", detect_multimodal_attention),
    }

    variant_name, detect_fn = variant_map[variant]
    print(f"[detect] Using variant: {variant_name}")

    mask = detect_fn(
        motion, flow, rms, flux,
        dt=dt, threshold_k=threshold_k,
    )

    active_pct = mask.sum() / len(mask) * 100
    print(f"[detect] {active_pct:.1f}% of video flagged as dynamic")

    # ============================================================== #
    #  STEP 4 – Temporal Post-Processing
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  STEP 4 / 5  –  Temporal Post-Processing")
    print("=" * 60)

    intervals, clip_dur, raw_intervals = build_extraction_map(
        mask, common_ts, total_dur,
        buffer_sec=buffer_sec,
        merge_gap_sec=merge_gap_sec,
        target_sec=target_sec,
    )

    print(f"[temporal] {len(raw_intervals)} raw segments → "
          f"{len(intervals)} final segments")
    print(f"[temporal] Total clip duration: {clip_dur:.1f}s "
          f"(target: {target_sec:.0f}s)")

    # ============================================================== #
    #  STEP 5 – Output
    # ============================================================== #
    print("\n" + "=" * 60)
    print("  STEP 5 / 5  –  Output Generation")
    print("=" * 60)

    # --- Composite signal for plotting ---
    w = {"motion": 0.30, "flow": 0.30, "rms": 0.20, "flux": 0.20}
    composite = (w["motion"] * motion + w["flow"] * flow
                 + w["rms"] * rms + w["flux"] * flux)
    threshold = adaptive_threshold_fast(composite, window_sec=30.0,
                                        dt=dt, k=threshold_k)

    # Diagnostic plot
    plot_path = os.path.join(output_dir, f"{base}_diagnostics.png")
    plot_diagnostics(
        timestamps=common_ts,
        motion=motion, flow=flow, rms=rms, flux=flux,
        composite=composite, threshold=threshold,
        mask=mask, intervals=intervals, raw_intervals=raw_intervals,
        video_duration=total_dur,
        output_path=plot_path,
        variant_name=variant_name,
    )

    # Export clip
    clip_path = os.path.join(output_dir, f"{base}_highlight.mp4")
    export_clip(video_path, intervals, clip_path)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Clip   : {clip_path}")
    print(f"  Plot   : {plot_path}")
    print(f"  Length  : {clip_dur:.1f}s from {total_dur:.1f}s source")
    print(f"{'=' * 60}\n")

    return intervals, clip_dur


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Highlight Extraction – automatically "
                    "extract ~1-min highlight clips from long-form video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection Variants:
  heuristic  (A)  Weighted combination of all channels.
                  Best for unstructured / diverse content (DEFAULT).
  surprisal  (B)  Information-theoretic: flags statistically unlikely
                  moments.  Good for uniform-background content.
  attention  (C)  Data-driven channel weighting via gradient-based
                  attention.  Good for multi-source content.

Examples:
  python extract_highlights.py lecture.mp4
  python extract_highlights.py game.mp4 --variant attention --target 90
  python extract_highlights.py vlog.mp4 --threshold 2.0 --buffer 3
        """,
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--variant", choices=["heuristic", "surprisal", "attention"],
        default="heuristic",
        help="Detection variant (default: heuristic)",
    )
    parser.add_argument(
        "--target", type=float, default=60.0,
        help="Target clip length in seconds (default: 60)",
    )
    parser.add_argument(
        "--buffer", type=float, default=5.0,
        help="Post-event safety buffer in seconds (default: 5)",
    )
    parser.add_argument(
        "--merge-gap", type=float, default=3.0,
        help="Max gap between segments to merge (default: 3s)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.5,
        help="Adaptive threshold sensitivity k (default: 1.5). "
             "Lower = more segments; higher = fewer, stronger peaks only.",
    )
    parser.add_argument(
        "--sample-fps", type=float, default=2.0,
        help="Video analysis sample rate (default: 2 fps)",
    )
    parser.add_argument(
        "--sg-window", type=int, default=15,
        help="Savitzky-Golay filter window (default: 15, must be odd)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: same as input video)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        video_path=args.video,
        variant=args.variant,
        target_sec=args.target,
        buffer_sec=args.buffer,
        merge_gap_sec=args.merge_gap,
        threshold_k=args.threshold,
        sample_fps=args.sample_fps,
        sg_window=args.sg_window,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
