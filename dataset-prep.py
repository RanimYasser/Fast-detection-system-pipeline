import os
import cv2
import serial
from datetime import datetime
from Control.camera_manger import SoftTriggerGrabber

# ===== Config =====
PORT = "COM6"          # ← change if needed
BAUD = 115200
SAVE_DIR = os.path.join(os.getcwd(), "top dataset")
JPEG_QUALITY = 95
FIRE_TIMEOUT = 0.35     # seconds (adjust if you occasionally miss frames)

# Your Allied Vision camera ID (example: "DEV_1AB22C00C5A5")
av_ids = SoftTriggerGrabber.discover_av_cameras()
if not av_ids:
    print("No AV cameras found. Exiting.")

cam_id = av_ids[0] if len(av_ids) >= 1 else None


def safe_cam(cam_id: str) -> str:
    return cam_id.replace(":", "_").replace("/", "_").replace("\\", "_")

def unique_name(cam_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{ts}_{safe_cam(cam_id)}.jpg"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ensure_dir(SAVE_DIR)
    g = SoftTriggerGrabber(cam_id, pixel_format='BGR8')
    try:
        with serial.Serial(PORT, BAUD, timeout=None) as ser:
            print("Waiting for switch press...")
            for raw in ser:  # blocks until a line arrives
                line = raw.decode(errors="ignore").strip().lower()
                if line == "d3: pressed":
                    img = g.fire(timeout=FIRE_TIMEOUT)
                    if img is None:
                        print("Trigger timeout (no frame).")
                        continue

                    fname = unique_name(cam_id)
                    fpath = os.path.join(SAVE_DIR, fname)

                    # Safer write via imencode → bytes (Windows-friendly)
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not ok:
                        print("imencode failed; skipping.")
                        continue
                    try:
                        with open(fpath, "wb") as f:
                            f.write(buf.tobytes())
                        print(f"Saved: {fpath}")
                    except Exception as e:
                        print(f"Save error: {e}")
    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        try:
            g.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
