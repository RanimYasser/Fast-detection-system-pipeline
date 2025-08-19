import time
import os
from datetime import datetime
import cv2

LOG_FILE = "run_log.txt"
BOX_INFO_FILE = "box_info.txt"

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe = (msg or "").replace("→", "->")
    line = f"[{ts}] {safe}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")

def save_box_info(box_info):
    with open(BOX_INFO_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"{box_info}\n")
    print(f"[save_box_info] {box_info}")

def _assert_file(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at: {path}")
    

def try_open_camera(index: int): #remove
    """Try common Windows backends for reliability; prefer DSHOW, set small buffers."""
    for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            # keep latency low
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap
        cap.release()
    return None

