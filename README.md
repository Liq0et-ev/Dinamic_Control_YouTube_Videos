# Dynamic Control YouTube Videos — Batch Highlight Extraction System

Automatically extracts ~1-minute highlight clips from **multiple** long-form videos (30+ min) by analysing visual and audio dynamics. Supports batch processing with case-based file selection.

## How It Works

```
SOURCE_DIR (your raw videos)
        │
        ▼
┌──────────────────────────┐
│  Directory Scan           │  Find all .mp4, .avi, .mkv, .mov, etc.
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Case Selection           │  all / single / custom list / range
└──────────┬───────────────┘
           ▼
   ┌───── FOR EACH VIDEO ─────┐
   │                           │
   │  1. Visual features       │  motion intensity + optical flow
   │  2. Audio features        │  RMS energy + spectral flux
   │  3. DSP pipeline          │  normalize → SG-smooth → derivatives
   │  4. Detection             │  adaptive threshold + onset detection
   │  5. Temporal processing   │  +5s buffer → merge → ~60s target
   │  6. Export clips          │  video_name_highlight_1.mp4, _2.mp4...
   │  7. Diagnostic plot       │  PNG with signal analysis
   │                           │
   └───── gc.collect() ────────┘
        │
        ▼
TARGET_DIR/
  ├── video_A/
  │   ├── video_A_highlight_1.mp4
  │   ├── video_A_highlight_2.mp4
  │   └── video_A_diagnostics.png
  ├── video_B/
  │   └── ...
  └── _processing_log.json
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit SOURCE_DIR and TARGET_DIR in main.py (top of file)
#    Or use --source-dir and --target-dir flags

# 3. Place your video files in SOURCE_DIR, then run:
python main.py
```

## Case Selection (Choosing Which Videos to Process)

| Command | What It Does |
|---|---|
| `python main.py` | **Interactive** — shows numbered list, asks for your choice |
| `python main.py --case all` | Process **every** video in SOURCE_DIR |
| `python main.py --case 3` | Process only video **#3** |
| `python main.py --case 1,3,5` | Process videos **#1, #3, and #5** |
| `python main.py --case 2-5` | Process videos **#2 through #5** |
| `python main.py --case 1,3-5,8` | Mixed: **#1, #3, #4, #5, #8** |

### Interactive Mode Example

```
  ┌─────┬────────────────────────────────────────┬──────────┐
  │  #  │  Filename                              │  Status  │
  ├─────┼────────────────────────────────────────┼──────────┤
  │   1 │  lecture_monday.mp4                    │    new   │
  │   2 │  football_highlights.mp4               │   DONE   │
  │   3 │  conference_keynote.mkv                │    new   │
  └─────┴────────────────────────────────────────┴──────────┘

  Your selection: 1,3
```

## Detection Variants

| Variant | Flag | Best For |
|---|---|---|
| **(A) Weighted Heuristic** | `--variant heuristic` | General / diverse content **(DEFAULT)** |
| **(B) Surprisal** | `--variant surprisal` | Uniform backgrounds, surveillance footage |
| **(C) Multimodal Attention** | `--variant attention` | Multi-source content, varying channels |

**Why Heuristic is the default:** It needs no training data, weights are transparent and tuneable, and the adaptive threshold handles domain shift across varied content types automatically.

## All Options

```
python main.py [OPTIONS]

Selection:
  --case SELECTION      'all', single #, comma list, or range (default: interactive)
  --force               Re-process videos even if already completed

Detection:
  --variant VARIANT     heuristic | surprisal | attention (default: heuristic)
  --threshold FLOAT     Sensitivity k — lower = more clips (default: 1.5)
  --target SECONDS      Target clip length per video (default: 60)
  --buffer SECONDS      Post-event safety buffer (default: 5)
  --merge-gap SECONDS   Max gap to merge segments (default: 3)

Processing:
  --sample-fps FLOAT    Analysis sample rate (default: 2 fps)
  --sg-window INT       Savitzky-Golay window size (default: 15)

Paths:
  --source-dir PATH     Override SOURCE_DIR
  --target-dir PATH     Override TARGET_DIR
```

## Output Structure

Each processed video gets its own subfolder in TARGET_DIR:

```
exported_highlights/
├── lecture_monday/
│   ├── lecture_monday_highlight_1.mp4    ← individual clip segments
│   ├── lecture_monday_highlight_2.mp4
│   ├── lecture_monday_highlight_3.mp4
│   └── lecture_monday_diagnostics.png    ← signal analysis plot
├── football_game/
│   ├── football_game_highlight_1.mp4
│   └── football_game_diagnostics.png
└── _processing_log.json                  ← tracks completed files
```

## Processing Log

A persistent `_processing_log.json` in TARGET_DIR tracks which files have been processed. Re-running the script automatically **skips completed files** unless you pass `--force`.

## Examples

```bash
# Process all videos with default settings
python main.py --case all

# Process specific videos with attention-based detection
python main.py --case 1,3,5 --variant attention

# More sensitive detection, longer clips
python main.py --case all --threshold 1.0 --target 90

# Re-process everything from scratch
python main.py --case all --force

# Custom source/target directories
python main.py --source-dir /videos/raw --target-dir /videos/highlights --case all
```

## Requirements

- Python 3.10+
- FFmpeg (must be on PATH)
- Libraries: OpenCV, Librosa, SciPy, NumPy, Matplotlib, MoviePy

## Project Structure

```
main.py                 Batch orchestrator, CLI, case selection
video_features.py       Visual feature extraction (motion + optical flow)
audio_features.py       Audio feature extraction (RMS + spectral flux)
dsp_pipeline.py         Signal normalization, SG-filtering, adaptive thresholds
detectors.py            3 detection algorithms (Heuristic / Surprisal / Attention)
temporal.py             Buffer logic, interval merging, target-length adjustment
diagnostics.py          Matplotlib diagnostic plot generation
processing_log.py       Persistent JSON log for batch tracking
requirements.txt        Python dependencies
```
