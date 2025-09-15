#!/usr/bin/env python3
# log_boxes_headless.py
#
# Headless runner that:
#  - starts your Fast Detection pipeline (no UI windows)
#  - reads messages from info_q (emitted when a Box is saved)
#  - logs each box to CSV, JSONL, and TXT under captures/logs/
#  - handles Ctrl+C clean shutdown

import os, sys, csv, json, signal, multiprocessing as mp
from datetime import datetime

# ---------- locate your project & import pipeline helpers ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your pipeline functions from Control/pipeline.py
from Model.newcomm2 import start_pipeline, stop_pipeline, LABEL_CAM_INDEX

# ---------- logging setup ----------
LOG_DIR = os.path.join("captures", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DATE_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH   = os.path.join(LOG_DIR, f"boxes_{DATE_STAMP}.csv")
JSONL_PATH = os.path.join(LOG_DIR, f"boxes_{DATE_STAMP}.jsonl")
TXT_PATH   = os.path.join(LOG_DIR, f"boxes_{DATE_STAMP}.txt")

CSV_FIELDS = [
    "ts", "event", "id",
    "brand", "flavor", "capacity", "product_type",
    "barcode", "expire", "status", "reason"
]

csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
csv_writer.writeheader()

jsonl_file = open(JSONL_PATH, "a", encoding="utf-8")
txt_file   = open(TXT_PATH, "a", encoding="utf-8")

def log_box(msg: dict):
    """Append one row to CSV/JSONL/TXT."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    row = {
        "ts": ts,
        "event": msg.get("event", ""),
        "id": msg.get("id", ""),
        "brand": msg.get("brand", ""),
        "flavor": msg.get("flavor", ""),
        "capacity": msg.get("capacity", ""),
        "product_type": msg.get("product_type", ""),
        "barcode": msg.get("barcode", ""),
        "expire": msg.get("expire", ""),
        "status": msg.get("status", ""),
        "reason": msg.get("reason", "")
    }
    # CSV
    csv_writer.writerow(row)
    csv_file.flush()
    # JSONL
    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    jsonl_file.flush()
    # TXT human-readable summary
    summary = (f"[{row['ts']}] {row['event']} | id={row['id']} | "
               f"{row['brand']} {row['flavor']} {row['capacity']} "
               f"| barcode={row['barcode']} | expire={row['expire']} "
               f"| status={row['status']} ({row['reason']})\n")
    txt_file.write(summary)
    txt_file.flush()
    # Also echo to console
    print(summary, end="")

# ---------- main ----------
def main():
    # Windows needs spawn; Linux/mac ok with default but we enforce for consistency
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"[Headless] Logs ->\n  CSV  : {CSV_PATH}\n  JSONL: {JSONL_PATH}\n  TXT  : {TXT_PATH}")

    # Start pipeline (headless mode if supported)
    try:
        pipe = start_pipeline(label_cam_index=LABEL_CAM_INDEX, batch_name="Orange Batch", headless=True)
    except TypeError:
        pipe = start_pipeline(label_cam_index=LABEL_CAM_INDEX, batch_name="Orange Batch")

    print("[Headless] Pipeline started. Press Ctrl+C to stop.")

    stopping = False
    def _stop(*_):
        nonlocal stopping
        if not stopping:
            stopping = True
            print("\n[Headless] Stopping …")
            try:
                stop_pipeline(pipe)
            except Exception:
                pass

    # Clean shutdown on Ctrl+C/terminate
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    info_q = pipe.info_q

    # Main loop: consume info_q and log
    while not stopping:
        try:
            msg = info_q.get(timeout=0.5)
            if isinstance(msg, dict) and msg.get("event") == "box_saved":
                log_box(msg)
        except Exception:
            # timeout or benign hiccup
            pass

    # Close files
    for f in (csv_file, jsonl_file, txt_file):
        try:
            f.close()
        except Exception:
            pass
    print("[Headless] Stopped cleanly.")

if __name__ == "__main__":
    main()
