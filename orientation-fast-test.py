# trigger_capture_barcode_d2_only.py
import serial
from ultralytics import YOLO
from Control.camera_manger2 import SoftTriggerGrabber
from Model.utils import log

# --- config ---
MODEL_PATH   = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_barcode.pt"
SERIAL_PORT  = "COM6"          # Arduino/Nano port
BAUD_RATE    = 115200
CAMERA_ID    = "DEV_1AB22C00C5A5"  # Allied Vision device id
CONF_THRESH  = 0.35
TRIGGER_TIMEOUT_S = 0.6        # wait time for frame after trigger

def main():
    # Load model
    model = YOLO(MODEL_PATH)

    # Camera (soft-trigger)
    grabber = SoftTriggerGrabber(CAMERA_ID)
    log(f"[Barcode] Using SoftTriggerGrabber id={CAMERA_ID}")

    # Serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Listening for D2: PRESSED on {SERIAL_PORT} ... (Ctrl+C to exit)")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line.lower() != "d2: pressed":
                continue  # ignore everything except D2 presses

            # --- Capture exactly one frame ---
            frame = grabber.fire(timeout=TRIGGER_TIMEOUT_S)
            if frame is None:
                print("False")  # timeout = treat as not detected
                continue

            # --- YOLO detect ---
            res = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
            print("True" if len(res.boxes) > 0 else "False")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            grabber.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
