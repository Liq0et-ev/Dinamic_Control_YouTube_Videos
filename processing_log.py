"""
Processing Log — Persistent Memory
====================================
Tracks which video files have been successfully processed so that
re-runs skip already-completed files (unless forced).

Stores a JSON log in the TARGET_DIR:
    {
        "video_name.mp4": {
            "status": "completed",
            "timestamp": "2026-03-31T14:22:00",
            "source_path": "/path/to/source/video_name.mp4",
            "clips_exported": 3,
            "total_clip_duration": 58.4
        }
    }
"""

import json
import os
from datetime import datetime


LOG_FILENAME = "_processing_log.json"


def _log_path(target_dir: str) -> str:
    return os.path.join(target_dir, LOG_FILENAME)


def load_log(target_dir: str) -> dict:
    """Load existing processing log, or return empty dict."""
    path = _log_path(target_dir)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_log(target_dir: str, log: dict):
    """Write the processing log to disk."""
    path = _log_path(target_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def mark_completed(target_dir: str, filename: str, source_path: str,
                   clips_exported: int, total_clip_duration: float):
    """Record a successful processing run."""
    log = load_log(target_dir)
    log[filename] = {
        "status": "completed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_path": source_path,
        "clips_exported": clips_exported,
        "total_clip_duration": round(total_clip_duration, 1),
    }
    save_log(target_dir, log)


def mark_failed(target_dir: str, filename: str, source_path: str, error: str):
    """Record a failed processing run."""
    log = load_log(target_dir)
    log[filename] = {
        "status": "failed",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_path": source_path,
        "error": str(error),
    }
    save_log(target_dir, log)


def is_completed(target_dir: str, filename: str) -> bool:
    """Check if a file has already been successfully processed."""
    log = load_log(target_dir)
    entry = log.get(filename)
    return entry is not None and entry.get("status") == "completed"


def print_log_summary(target_dir: str):
    """Print a summary of the processing log."""
    log = load_log(target_dir)
    if not log:
        print("  (no processing history)")
        return

    completed = sum(1 for v in log.values() if v.get("status") == "completed")
    failed = sum(1 for v in log.values() if v.get("status") == "failed")
    print(f"  Previously processed: {completed} completed, {failed} failed")
    for name, info in log.items():
        status = info.get("status", "unknown")
        symbol = "OK" if status == "completed" else "FAIL"
        print(f"    [{symbol}] {name} ({info.get('timestamp', '?')})")
