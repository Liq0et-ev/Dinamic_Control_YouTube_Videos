# Dynamic Control YouTube Videos — Highlight Extraction System

Automatically extracts ~1-minute highlight clips from long-form video (30+ minutes) by analysing **visual and audio dynamics** together.

## How It Works

```
Input Video (30+ min)
        │
        ▼
┌──────────────────────┐
│  Feature Extraction   │  4 channels: motion, optical flow, RMS energy, spectral flux
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Signal Processing    │  Normalize → Savitzky-Golay smooth → Derivatives
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Detection            │  3 variants: Weighted Heuristic / Surprisal / Attention
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Temporal Processing  │  Median filter → +5s buffer → Merge → Target ~60s
└──────────┬───────────┘
           ▼
  Highlight clip (MP4) + Diagnostic plot (PNG)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on any video file
python extract_highlights.py your_video.mp4
```

This produces two files next to your video:
- `your_video_highlight.mp4` — the extracted highlight clip
- `your_video_diagnostics.png` — visual breakdown of what the system detected

## Detection Variants

| Variant | Flag | Best For |
|---|---|---|
| **(A) Weighted Heuristic** | `--variant heuristic` | General / unstructured video (DEFAULT) |
| **(B) Surprisal** | `--variant surprisal` | Uniform backgrounds, surveillance-style footage |
| **(C) Multimodal Attention** | `--variant attention` | Multi-source content with varying dominant channels |

**Recommendation:** Variant A (Weighted Heuristic) is the best default for unstructured video because it needs no training data, the weights are transparent, and the per-channel adaptive threshold handles domain shift automatically.

## All Options

```
python extract_highlights.py video.mp4 [OPTIONS]

  --variant {heuristic,surprisal,attention}   Detection method (default: heuristic)
  --target SECONDS        Target clip length (default: 60)
  --buffer SECONDS        Post-event safety buffer (default: 5)
  --merge-gap SECONDS     Max gap to merge adjacent segments (default: 3)
  --threshold FLOAT       Sensitivity k — lower = more segments (default: 1.5)
  --sample-fps FLOAT      Analysis sample rate (default: 2 fps)
  --sg-window INT         Savitzky-Golay window size (default: 15, must be odd)
  --output-dir PATH       Output directory (default: same as input)
```

## Examples

```bash
# Sports game — use attention variant, allow 90s clip
python extract_highlights.py game.mp4 --variant attention --target 90

# Lecture — more sensitive detection
python extract_highlights.py lecture.mp4 --threshold 1.0

# Quick test — fewer segments, strict threshold
python extract_highlights.py vlog.mp4 --threshold 2.5 --target 30
```

## Requirements

- Python 3.10+
- FFmpeg (must be on PATH)
- Libraries: OpenCV, Librosa, SciPy, NumPy, Matplotlib, MoviePy

## Project Structure

```
extract_highlights.py   Main CLI entry point & orchestrator
video_features.py       Visual feature extraction (motion + optical flow)
audio_features.py       Audio feature extraction (RMS + spectral flux)
dsp_pipeline.py         Signal normalization, SG-filtering, adaptive thresholds
detectors.py            3 detection algorithms (Heuristic / Surprisal / Attention)
temporal.py             Buffer logic, interval merging, target-length adjustment
diagnostics.py          Matplotlib diagnostic plot generation
requirements.txt        Python dependencies
```
