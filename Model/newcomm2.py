# Model/communication.py
import os
import sys
import cv2
import time
import multiprocessing as mp
from datetime import datetime
from queue import Empty
from ultralytics import YOLO
import numpy as np
from collections import namedtuple
from typing import Optional
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent

from Control.ReadExcel import readExcel, findProduct, matchBarcode, product_from_parts
from Model.utils import log, save_box_info, _assert_file, try_open_camera
from Control.sensor import start_serial_listener
from Control.camera_manger2 import SoftTriggerGrabber
# NOTE: Do NOT import onnx_infer here. We lazy-import it only if/when ONNX is used.

# =====================================================================================
# Config
# =====================================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# General label/track model
MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best.pt"
# Barcode ROI model
BARCODE_MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\bestBarcode.pt"

# Hardcoded expiry for pipeline test
TEST_DATE = "2025-12-31"

# Label camera index (OpenCV webcam)
LABEL_CAM_INDEX = 0
TRIGGER_X = 200  # px position to assign Box -> first track that crosses this line

# Where to save barcode captures
BARCODE_SAVE_DIR = os.path.join("captures", "barcode")
os.makedirs(BARCODE_SAVE_DIR, exist_ok=True)

# === GPU usage switches ===
# Use DirectML (AMD GPU) for BARCODE stage (ONNX). Keep LABEL stage on Ultralytics CPU for tracker.
USE_ONNX_DML = True
USE_ONNX_DML_FOR_LABEL = False  # keeping False preserves model.track()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ONNX_MODEL_PATH   = os.path.join(ROOT, "Yolo-models", "best.onnx")
ONNX_BARCODE_PATH = os.path.join(ROOT, "Yolo-models", "bestBarcode.onnx")

# =====================================================================================
# Project imports (relative safe import)
# =====================================================================================
try:
    from Model.box import Box
    from Model.detection import Detect
except ModuleNotFoundError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT_DIR)
    from Model.box import Box
    from Model.detection import Detect


def initialize_label_model():
    """
    LABEL stage needs Ultralytics' tracker. Keep it on CPU to avoid CUDA/AMD issues.
    """
    if USE_ONNX_DML_FOR_LABEL:
        # Advanced path (not used here): ONNX + custom tracker
        from Model.onnx_infer import new_dml_session
        return new_dml_session(ONNX_MODEL_PATH), None
    else:
        _assert_file(MODEL_PATH, "MODEL_PATH")
        model = YOLO(MODEL_PATH)
        return model, None


def initialize_barcode_model():
    """
    BARCODE stage: use ONNX + DirectML (AMD GPU) when enabled, else Ultralytics CPU.
    """
    if USE_ONNX_DML:
        from Model.onnx_infer import new_dml_session  # lazy import
        return new_dml_session(ONNX_BARCODE_PATH)
    else:
        _assert_file(BARCODE_MODEL_PATH, "BARCODE_MODEL_PATH")
        return YOLO(BARCODE_MODEL_PATH)

# =====================================================================================
# Stage 1a: Barcode Capture (triggered by switch) — AV + software trigger + SAVE + bytes
# =====================================================================================
def barcode_capture_process(event_queue: mp.Queue, barcode_image_queue: mp.Queue, cam_id: str):
    log(f"[Barcode Capture] Using AV camera id={cam_id}")
    g = SoftTriggerGrabber(cam_id, pixel_format='BGR8')
    log(f"[Barcode Capture] Saving to: {BARCODE_SAVE_DIR}")
    log("[Barcode Capture] Ready. Waiting for press...")

    try:
        while True:
            try:
                event, event_time = event_queue.get(timeout=1)
                if event != "switch pressed":
                    continue

                # Trigger a new exposure and get the frame
                img = g.fire(timeout=0.3)
                if img is None:
                    log("[Barcode Capture] Trigger timeout (no frame).")
                    continue

                # Save with timestamp and cam id
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                fname = f"{ts}_{cam_id.replace(':','_').replace('/','_')}.jpg"
                fpath = os.path.join(BARCODE_SAVE_DIR, fname)
                try:
                    ok = cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if ok:
                        log(f"[Barcode Capture] Saved {fname}")
                    else:
                        log(f"[Barcode Capture] Failed to save {fname}")
                except Exception as e:
                    log(f"[Barcode Capture] Save error: {e}")

                # Put JPEG bytes on queue (Windows-picklable)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    log("[Barcode Capture] imencode failed")
                    continue
                barcode_image_queue.put(buf.tobytes())

            except Empty:
                time.sleep(0.02)
            except Exception as e:
                log(f"[Barcode Capture Error] {e}")
                time.sleep(0.05)
    finally:
        try:
            g.close()
        except Exception:
            pass

# =====================================================================================
# Stage 1b: Barcode Process (detect ROI + decode -> create Box -> queue to Expire)
# =====================================================================================
def barcode_process(barcode_image_queue: mp.Queue, box_q_B2E: mp.Queue):
    """
    Barcode stage. Prefer ONNX+DirectML (AMD GPU). If DirectML init/import fails,
    automatically fall back to Ultralytics CPU so the process never dies.
    """
    use_onnx = False
    names = None

    if USE_ONNX_DML:
        try:
            # Lazy import so normal CPU path works even if ORT is missing
            from Model.onnx_infer import new_dml_session, run_yolo_onnx
            try:
                model = new_dml_session(ONNX_BARCODE_PATH)  # may raise if ORT/DML is broken
                names = {0: "barcode"}  # adjust if your model has more classes
                use_onnx = True
                log("[Barcode Process] Using ONNX Runtime DirectML (GPU).")
            except Exception as e:
                log(f"[Barcode Process] DirectML init failed: {e} -> falling back to Ultralytics CPU.")
                _assert_file(BARCODE_MODEL_PATH, "BARCODE_MODEL_PATH")
                model = YOLO(BARCODE_MODEL_PATH)
                names = getattr(model, "names", {0: "barcode"})
                use_onnx = False
        except Exception as e:
            # Even the import of onnx_infer failed
            log(f"[Barcode Process] onnx_infer import error: {e} -> falling back to Ultralytics CPU.")
            _assert_file(BARCODE_MODEL_PATH, "BARCODE_MODEL_PATH")
            model = YOLO(BARCODE_MODEL_PATH)
            names = getattr(model, "names", {0: "barcode"})
            use_onnx = False
    else:
        _assert_file(BARCODE_MODEL_PATH, "BARCODE_MODEL_PATH")
        model = YOLO(BARCODE_MODEL_PATH)
        names = getattr(model, "names", {0: "barcode"})
        use_onnx = False

    detect = Detect()
    next_box_id = 0

    while True:
        data = barcode_image_queue.get()  # blocking
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            log("[Barcode Process] imdecode failed")
            continue

        try:
            if use_onnx:
                # returns [{'xyxy':(x1,y1,x2,y2), 'conf':float, 'cls':int}, ...]
                from Model.onnx_infer import run_yolo_onnx
                dets = run_yolo_onnx(model, img, conf_thr=0.8)
            else:
                ul_results = model(img, conf=0.8, device="cpu", verbose=False)
                r0 = ul_results[0]
                dets = []
                if len(r0.boxes):
                    for b in r0.boxes:
                        (x1, y1, x2, y2) = map(float, b.xyxy[0])
                        dets.append({
                            "xyxy": (x1, y1, x2, y2),
                            "conf": float(b.conf[0]),
                            "cls": int(b.cls[0])
                        })
        except Exception as e:
            log(f"[Barcode YOLO Error] {e}")
            dets = []

        new_box = Box()
        new_box.set_id(next_box_id)
        next_box_id += 1

        if not dets:
            new_box.set_barcode("inverted")
        else:
            decoded = None
            for d in dets:
                cls = d["cls"]
                cls_name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
                if cls_name.lower() == "barcode":
                    x1, y1, x2, y2 = map(int, d["xyxy"])
                    roi = img[y1:y2, x1:x2]
                    try:
                        decoded = detect.barcode_detection(roi)
                    except Exception as e:
                        log(f"[Barcode Decode Error] {e}")
                        decoded = None
                    break

            new_box.set_barcode(decoded if decoded else None)

        box_q_B2E.put(new_box)
        log(f"[Barcode Process] Box #{new_box.id} -> box_q_B2E")

# =====================================================================================
# Stage 2a: Expire Capture (wait for Box -> capture expiry cam -> send work)
# =====================================================================================
def expire_capture_process(box_q_B2E: mp.Queue, expire_work_q: mp.Queue):
    while True:
        try:
            current_box = box_q_B2E.get(timeout=2)
            expire_work_q.put((current_box, None))
        except Empty:
            time.sleep(0.02)

# =====================================================================================
# Stage 2b: Expire Process (set TEST_DATE) -> queue to Label
# =====================================================================================
def expire_process(expire_work_q: mp.Queue, box_q_E2L: mp.Queue):
    while True:
        try:
            current_box, _img = expire_work_q.get(timeout=2)  # unpack tuple
            current_box.set_expire_date(TEST_DATE)
            box_q_E2L.put(current_box)  # send only Box, not tuple
            log(f"[Expire Process] Box #{current_box.id} (expiry={TEST_DATE}) -> box_q_E2L")
        except Empty:
            time.sleep(0.05)
        except Exception as e:
            log(f"[Expire Process Error] {e}")
            time.sleep(0.1)

# =====================================================================================
# Label Worker helpers
# =====================================================================================
def load_batch_data(batch_name: Optional[str]):
    """Load Excel batch file only if provided."""
    if not batch_name:
        return
    batch_file = f"{batch_name}.xlsx"
    readExcel(batch_file)  # module-level df in Control.ReadExcel


def recover_camera(camera_index: int):
    """Handle camera reopen if frame fails."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        return cap, ok, frame
    return cap, False, None


def assign_box(track_id, cx, box_q_E2L, assigned, seen_tracks, processed_ids, trigger_x=TRIGGER_X):
    """Assign a pending box to a detected track if it crosses trigger (one-shot)."""
    if track_id in processed_ids:
        return
    if (track_id not in seen_tracks) and (cx >= trigger_x):
        try:
            pending_box = box_q_E2L.get_nowait()
            assigned[track_id] = pending_box
            seen_tracks.add(track_id)
            log(f"[LabelWorker] Assigned Box #{pending_box.id} -> track {track_id}")
        except Empty:
            pass


def _norm_capacity_for_cap_check(capacity: str) -> str:
    # "1 L" -> "1l", "1000 ml" -> "1000ml"
    c = (capacity or "").strip().lower().replace(" ", "")
    return c


def process_detection(track_id, class_name, assigned, detect, info_q, processed_ids):
    if track_id in processed_ids:
        assigned.pop(track_id, None)
        return

    box = assigned.get(track_id)
    if box is None:
        return

    try:
        # Parse YOLO class into product parts
        parts = detect.label_detections(class_name)  # [brand, flavor, capacity, (opt) type]
        if not parts or len(parts) < 3:
            box.set_status("rejected")
            box.set_reason("parse_error")
        else:
            brand, flavor, capacity = parts[:3]
            box.set_brand(brand)
            box.set_flavor(flavor)
            box.set_capacity(capacity)
            if len(parts) >= 4:
                box.set_product_type(parts[3])

            # Decision logic (inside Box)
            box.evaluate_box()

        # Notify UI once
        info = box.get_box_info()
        save_box_info(info)
        try:
            info_q.put_nowait({
                "event": "box_saved",
                "id": box.id,
                "brand": box.brand or "-",
                "flavor": box.flavor or "-",
                "capacity": box.capacity or "-",
                "product_type": box.product_type or "-",
                "barcode": box.get_barcode() or "-",
                "expire": getattr(box, "expire_date", "") or "-",
                "status": box.get_status(),
                "reason": box.get_reason() or ""
            })
        except Exception:
            pass

    except Exception as e:
        log(f"[LabelWorker] Processing error: {e}")
    finally:
        assigned.pop(track_id, None)
        processed_ids.add(track_id)

# =====================================================================================
# Stage 3: Label (OpenCV webcam)
# =====================================================================================
def label_worker(
    camera_index: int,
    box_q_E2L: MPQueue,
    frame_q: MPQueue,
    info_q: MPQueue,
    stop_event: MPEvent,
    batch_name: Optional[str] = None
):
    """Main label worker process (tracks boxes and enriches them)."""
    time.sleep(1.5)
    cap = try_open_camera(camera_index)
    if not cap:
        log(f"[LabelWorker] Failed to open camera {camera_index}")
        return

    model, detect = initialize_label_model()
    if detect is None:
        detect = Detect()  # ensure we have a parser for label strings

    load_batch_data(batch_name)

    assigned, seen_tracks = {}, set()
    processed_ids = set()  # one-shot processed track IDs
    device = "cpu"         # keep CPU for tracking (stable on AMD/Windows)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    log(f"[LabelWorker] Camera {camera_index} started.")

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            cap, ok, frame = recover_camera(camera_index)
            if not ok:
                log("[LabelWorker] Camera recovery failed.")
                break

        # Run YOLO tracking (Ultralytics, CPU)
        try:
            results = model.track(source=frame, conf=0.7, persist=True, device=device, verbose=False)
        except Exception as e:
            log(f"[LabelWorker] YOLO track error, retrying CPU: {e}")
            results = model.track(source=frame, conf=0.7, persist=True, device="cpu", verbose=False)
            device = "cpu"

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            names = getattr(results[0], "names", None) or getattr(model, "names", {})

            for i in range(len(boxes)):
                try:
                    track_id = int(boxes.id[i].item())
                    class_id = int(boxes.cls[i].item())
                    class_name = names[class_id] if isinstance(names, dict) else str(class_id)
                    if class_id in (13, 14):
                        continue

                    # Draw rectangle for debugging
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                    cx = (x1 + x2) // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, max(0, y1 - 8)),
                                FONT, 0.55, (0, 0, 255), 1)

                    # Box assignment (one-shot guard)
                    assign_box(track_id, cx, box_q_E2L, assigned, seen_tracks, processed_ids)

                    # Process once assigned (and never again)
                    if track_id in assigned:
                        process_detection(track_id, class_name, assigned, detect, info_q, processed_ids)

                except Exception as e:
                    log(f"[LabelWorker] Error: {e}")
                    continue

        # Push frame for UI display
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                if frame_q.full():
                    _ = frame_q.get_nowait()  # drop stale
                frame_q.put_nowait(buf.tobytes())
        except Exception:
            pass

    cap.release()
    log("[LabelWorker] Stopped.")

# =====================================================================================
# Pipeline start/stop helpers for the UI
# =====================================================================================
Pipeline = namedtuple("Pipeline", ["procs", "serial_thread", "frame_q", "info_q", "stop_event"])

def start_pipeline(label_cam_index: int = LABEL_CAM_INDEX,
                   frame_q: Optional[MPQueue] = None,
                   info_q: Optional[MPQueue] = None,
                   batch_name: Optional[str] = None):

    barcode_event_queue = mp.Queue()
    barcode_image_queue = mp.Queue()
    box_q_B2E = mp.Queue()
    expire_work_q = mp.Queue()
    box_q_E2L = mp.Queue()

    # UI streams
    frame_q = frame_q or mp.Queue(maxsize=2)
    info_q = info_q or mp.Queue(maxsize=64)
    stop_event = mp.Event()

    # Serial thread
    serial_thread = start_serial_listener(barcode_event_queue)

    # Discover Allied Vision cameras
    av_ids = SoftTriggerGrabber.discover_av_cameras()
    if not av_ids:
        log("[Comm] No Allied Vision cameras found! Barcode/Expire capture will not start.")

    barcode_cam_id = av_ids[0] if len(av_ids) >= 1 else None

    procs = []

    # Barcode capture+process
    if barcode_cam_id:
        procs.append(mp.Process(target=barcode_capture_process,
                                args=(barcode_event_queue, barcode_image_queue, barcode_cam_id),
                                daemon=True))
        procs.append(mp.Process(target=barcode_process,
                                args=(barcode_image_queue, box_q_B2E),
                                daemon=True))
    else:
        log("[Comm] Skipping barcode capture/process (no AV camera).")

    # Expiry capture + processing
    procs.append(mp.Process(target=expire_capture_process,
                            args=(box_q_B2E, expire_work_q),
                            daemon=True))
    procs.append(mp.Process(target=expire_process,
                            args=(expire_work_q, box_q_E2L),
                            daemon=True))

    # Label worker
    procs.append(mp.Process(
        target=label_worker,
        args=(label_cam_index, box_q_E2L, frame_q, info_q, stop_event, batch_name),
        daemon=True))

    for p in procs:
        p.start()

    log("[Comm] Pipeline started (AV-triggered capture + JPEG queues).")
    return Pipeline(procs=procs, serial_thread=serial_thread, frame_q=frame_q, info_q=info_q, stop_event=stop_event)

def stop_pipeline(pipeline: Pipeline):
    try:
        pipeline.stop_event.set()
    except Exception:
        pass
    for p in pipeline.procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    for p in pipeline.procs:
        try:
            p.join(timeout=2.0)
        except Exception:
            pass
    log("[Comm] Pipeline stopped.")
