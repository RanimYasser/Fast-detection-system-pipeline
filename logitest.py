# pip install ultralytics opencv-python
import cv2
import time
import torch
from collections import deque
from ultralytics import YOLO

# === Config ===
MODEL_PATH    = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_label.pt"
CAMERA_INDEX  = 0           # change to your webcam index
IMG_SIZE      = 640         # inference size (match your training if you want)
CONF_THRESH   = 0.7        # confidence threshold
WIN_NAME      = "YOLO Live"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(MODEL_PATH)
    model.to(device)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")



    # Simple FPS smoother
    times = deque(maxlen=30)
    last_t = time.perf_counter()

    print("Press 'q' or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        # YOLO inference
        t0 = time.perf_counter()
        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            verbose=False,
            device=device
        )
        t1 = time.perf_counter()

        r = results[0]
        # r.speed contains {'preprocess': ms, 'inference': ms, 'postprocess': ms}
        spd = getattr(r, "speed", {})
        pre_ms  = float(spd.get("preprocess", 0.0))
        inf_ms  = float(spd.get("inference", 0.0))
        post_ms = float(spd.get("postprocess", 0.0))
        total_ms = pre_ms + inf_ms + post_ms

        # Print per-frame inference timing
        print(f"Inference: {inf_ms:.2f} ms  (pre {pre_ms:.2f} | post {post_ms:.2f} | total {total_ms:.2f})")

        # Draw detections using Ultralytics' built-in annotator (boxes + labels)
        annotated = r.plot()  # returns a BGR numpy array with drawn boxes/labels

        # FPS calculation (smoothed)
        now = time.perf_counter()
        times.append(now - last_t)
        last_t = now
        fps = (len(times) / sum(times)) if times else 0.0

        # Overlay timing/FPS text
        overlay = f"{inf_ms:.1f} ms (inf) | {total_ms:.1f} ms (total) | {fps:.1f} FPS"
        cv2.putText(annotated, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high") if torch.cuda.is_available() else None
    main()
