# sensor_capture_test.py
# pip install pyserial opencv-python
import os, cv2, time, json, sys, datetime as dt
import serial
from camera_grabber import SoftTriggerGrabber  # same folder, or fix import path

# ---------- Config ----------
PORT = "COM7"             # <-- your Mega's COM port
BAUD = 115200

CAMERA_ID = "DEV_1AB22C00C5A5"   # <-- your Allied Vision ID
USE_HW_TRIGGER = False           # False => Software trigger; True => Hardware Line1 trigger
CAPTURE_DIR = r"captures\triggers"
LOCKOUT_MS = 150
IMG_JPEG_QUALITY = 95
ANNOTATE = True
TRIGGER_SENSORS = None           # e.g., {1} to only capture on sensor==1

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def mm_to_cm(mm):  return (mm or 0) / 10.0

def open_serial(port, baud):
    ser = serial.Serial(port, baud, timeout=0.2, rtscts=False, dsrdtr=False)
    time.sleep(2.0); ser.reset_input_buffer()
    return ser

def iter_json_lines(ser):
    buf = b""
    while True:
        chunk = ser.read(256)
        if not chunk:
            yield None; continue
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            s = line.decode('utf-8', 'ignore').strip()
            i, j = s.find('{'), s.rfind('}')
            if i != -1 and j != -1 and j > i:
                try: yield json.loads(s[i:j+1])
                except json.JSONDecodeError: pass

def annotate(frame, text):
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return frame

def save_frame(frame, sensor, d1_mm=None, d2_mm=None):
    day = dt.datetime.now().strftime("%Y-%m-%d")
    ensure_dir(os.path.join(CAPTURE_DIR, day))
    ts = dt.datetime.now().strftime("%H%M%S_%f")[:-3]
    fname = f"{ts}_S{sensor}_d1-{d1_mm or 0}mm_d2-{d2_mm or 0}mm.jpg"
    path = os.path.join(CAPTURE_DIR, day, fname)
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), IMG_JPEG_QUALITY])
    return path

def main():
    # --- serial ---
    try:
        ser = open_serial(PORT, BAUD)
        print(f"[Serial] Listening on {PORT} @ {BAUD} …")
    except serial.SerialException as e:
        print(f"[Serial Error] {e}"); sys.exit(1)

    # --- camera ---
    trig_src = 'Line1' if USE_HW_TRIGGER else 'Software'
    try:
        cam = SoftTriggerGrabber(
            CAMERA_ID,
            trigger_source=trig_src,   # 'Software' or 'Line1'
            pixel_format='BGR8',
            exposure_us=3000.0,
            gain_db=4.0,
            buffers=4
        )
        print(f"[Camera] Ready on {CAMERA_ID} (trigger_source={trig_src})")
    except Exception as e:
        print(f"[Camera Error] {e}"); sys.exit(1)

    hb = tr = 0
    t_stats = time.time()
    last_cap_ms = 0

    try:
        for msg in iter_json_lines(ser):
            if msg is None:
                continue

            evt = (msg.get("event") or "").lower()

            if evt == "hb":
                hb += 1
                d1 = msg.get("d1") or msg.get("d1_mm") or 0
                d2 = msg.get("d2") or msg.get("d2_mm") or 0
                near1 = msg.get("near1"); near2 = msg.get("near2")
                print(f"HB  d1={d1:4d} mm ({mm_to_cm(d1):4.1f} cm)  "
                      f"d2={d2:4d} mm ({mm_to_cm(d2):4.1f} cm)  near1={near1} near2={near2}")

            elif evt == "lidar_trigger":
                sensor = msg.get("sensor")

                if TRIGGER_SENSORS and sensor not in TRIGGER_SENSORS:
                    continue

                # lockout to avoid double fires
                now_ms = int(time.time() * 1000)
                if now_ms - last_cap_ms < LOCKOUT_MS:
                    continue
                last_cap_ms = now_ms

                tr += 1
                d1 = msg.get("d1") or msg.get("d1_mm") or 0
                d2 = msg.get("d2") or msg.get("d2_mm") or 0
                print(f"TRIGGER S{sensor}  d1={d1}mm  d2={d2}mm -> capture")

                # ---- capture exactly one frame ----
                t0 = time.perf_counter()
                if USE_HW_TRIGGER:
                    frame, meta = cam.wait_next(timeout=0.5)   # waiting for TTL on Line1
                else:
                    frame, meta = cam.fire(timeout=0.5)        # software trigger
                t1 = time.perf_counter()

                if frame is None:
                    print("[Capture] Timeout (no frame).")
                    continue

                cap_ms = (t1 - t0) * 1000.0
                info = f"S{sensor} d1={d1}mm({mm_to_cm(d1):.1f}cm) d2={d2}mm({mm_to_cm(d2):.1f}cm) id={meta['frame_id']} cap={cap_ms:.1f}ms"
                if ANNOTATE: annotate(frame, info)
                path = save_frame(frame, sensor, d1, d2)
                print(f"[Saved] {path}")

            # stats
            if time.time() - t_stats > 5:
                print(f"[stats] hb={hb} trig={tr} over {time.time() - t_stats:.1f}s")
                hb = tr = 0; t_stats = time.time()

    except KeyboardInterrupt:
        print("\n[Serial] Stopped by user.")
    finally:
        try: ser.close()
        except: pass
        try: cam.close()
        except: pass

if __name__ == "__main__":
    main()
