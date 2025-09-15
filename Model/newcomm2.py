import os
import sys
import cv2
import time
import multiprocessing as mp
import torch
from datetime import datetime
from queue import Empty
from ultralytics import YOLO
import numpy as np
from collections import namedtuple, deque
from typing import Optional
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from multiprocessing import Barrier

# --- add these two early ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Control.ReadExcel import readExcel, findProduct, matchBarcode, product_from_parts
from Model.utils import log, save_box_info, _assert_file, try_open_camera
from Control.sensor2 import start_serial_listener
from Control.camera_manger2 import SoftTriggerGrabber

# Config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_label.pt"
BARCODE_MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_barcode.pt"
TOP_MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_top.pt"
TEST_DATE = True
LABEL_CAM_INDEX = 0
TRIGGER_X = 200
BARCODE_SAVE_DIR = os.path.join("captures", "barcode")
os.makedirs(BARCODE_SAVE_DIR, exist_ok=True)
TOP_SAVE_DIR = os.path.join("captures", "top")
os.makedirs(TOP_SAVE_DIR, exist_ok=True)

# Project imports (relative safe import)
try:
    from Model.box import Box
    from Model.detection import Detect
except ModuleNotFoundError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
    sys.path.append(ROOT_DIR)
    from Model.box import Box
    from Model.detection import Detect

# Device picker (CUDA / DirectML / CPU)

def warmup_yolo_for_detect(model, device="cpu", imgsz=480):
    import numpy as np, cv2
    dummy = np.zeros((imgsz, imgsz, 3), np.uint8)
    # One detect forward
    _ = model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    # One more to stabilize memory allocator
    _ = model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

def prepare_yolo_model(model, device="cpu"):
    """Fuse Conv+BN, move to CUDA if available, and use FP16 on CUDA."""
    try:
        model.fuse()
    except Exception:
        pass
    if device == "cuda":
        try:
            model.to("cuda")
        except Exception:
            pass
        try:
            # Some Ultralytics versions expose the underlying model as .model
            model.model.half()
        except Exception:
            pass

def pick_device():
    # Ultralytics can't use 'dml' as a device string; sanitize env if present
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() == "dml":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Prefer CUDA if really available
    if torch.cuda.is_available():
        return "cuda"

    # NOTE: Even if torch-directml is installed, Ultralytics cannot take 'dml' here.
    # For reliability, use CPU. (DML option shown further below.)
    return "cpu"

DEVICE = pick_device()

# Stage 1a: Barcode Capture (triggered by switch) — AV + software trigger + SAVE + bytes

def barcode_capture_process(event_queue: mp.Queue, barcode_image_queue: mp.Queue, cam_id: str):
    #log(f"[Barcode Capture] Using AV camera id={cam_id}")
    g = SoftTriggerGrabber(cam_id, pixel_format='BGR8')
    #log(f"[Barcode Capture] Saving to: {BARCODE_SAVE_DIR}")
    #log("[Barcode Capture] Ready. Waiting for press...]")

    try:
        while True:
            try:
                evt, t_arduino_us, t_pc, delta_ms = event_queue.get()
                if evt != "triggered":
                    continue

                # Trigger a new exposure and get the frame
                img = g.fire(timeout=0.3)
                if img is None:
                    #log("[Barcode Capture] Trigger timeout (no frame).")
                    continue

                # Put JPEG bytes on queue (Windows-picklable)
                ok, buf = cv2.imencode(".jpg", img, [
                    cv2.IMWRITE_JPEG_QUALITY, 80,      # 95→80 is a big speed win
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0,      # keep OFF
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0    # keep OFF
                ])
                if not ok:
                    #log("[Barcode Capture] imencode failed")
                    continue

                barcode_image_queue.put(buf.tobytes())
                log(f"[barcode Capture] Saved image")

            except Empty:
                time.sleep(0.02)
            except Exception as e:
                #log(f"[Barcode Capture Error] {e}")
                time.sleep(0.05)
    finally:
        try:
            g.close()
        except Exception:
            pass

# Stage 1b: Barcode Process (detect ROI + decode -> create Box -> queue to Expire)

def barcode_process(barcode_image_queue: mp.Queue, box_q_B2E: mp.Queue, ready):
    _assert_file(BARCODE_MODEL_PATH, "BARCODE_MODEL_PATH")
    model = YOLO(BARCODE_MODEL_PATH)
    prepare_yolo_model(model, DEVICE)
    warmup_yolo_for_detect(model, DEVICE, imgsz=480)  # fused, FP16 on CUDA, imgsz=480
    ready.wait()
    detect = Detect()
    next_box_id = 0

    while True:
        # Receive JPEG bytes, decode to ndarray
        data = barcode_image_queue.get()  # blocking
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            #log("[Barcode Process] imdecode failed")
            continue

        log("[Barcode Process] Processing image...")
        try:
            results = model(img, conf=0.7, device=DEVICE, imgsz=480, verbose=False)[0]
        except Exception as e:
            #log(f"[Barcode YOLO Error] {e}")
            continue

        new_box = Box()
        new_box.set_id(next_box_id)
        next_box_id += 1
        log(f"[Barcode Process] Created Box #{new_box.id}")

        if len(results.boxes) == 0:
            new_box.set_barcode("inverted")
            log("No barcode class found in the image.")
        else:
            for b in results.boxes:
                cls = int(b.cls[0])
                cls_name = model.names[cls]
                if cls_name.lower() == "barcode":
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    roi = img[y1:y2, x1:x2]
                    try:
                        decoded = detect.barcode_detection(roi)
                    except Exception as e:
                        #log(f"[Barcode Decode Error] {e}")
                        decoded = None
                    if decoded:
                        log(f"Detected barcode: {decoded}")
                        new_box.set_barcode(decoded)
                        break
                    else:
                        log("No barcodes found in the image.")
                        new_box.set_barcode(None)

        box_q_B2E.put(new_box)
        log(f"[Barcode Process] Box #{new_box.id} -> box_q_B2E")

def initialize_top_model():
    _assert_file(TOP_MODEL_PATH, "TOP_MODEL_PATH")
    return YOLO(TOP_MODEL_PATH)

def top_capture_process(event_queue: mp.Queue, top_image_queue: mp.Queue, cam_id: str):
    #log(f"[top Capture] Using AV camera id={cam_id}")
    g = SoftTriggerGrabber(cam_id, pixel_format='BGR8')
    #log(f"[top Capture] Saving to: {TOP_SAVE_DIR}")
    #log("[top Capture] Ready. Waiting for press...]")

    try:
        while True:
            try:
                evt, t_arduino_us, t_pc, delta_ms = event_queue.get()
                if evt != "triggered":
                    continue

                # Trigger a new exposure and get the frame
                img = g.fire(timeout=0.3)
                if img is None:
                    log("[TOP Capture] Trigger timeout (no frame).")
                    continue

                # Put JPEG bytes on queue (Windows-picklable)
                ok, buf = cv2.imencode(".jpg", img, [
                    cv2.IMWRITE_JPEG_QUALITY, 80,      # 95→80 is a big speed win
                    cv2.IMWRITE_JPEG_OPTIMIZE, 0,      # keep OFF
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0    # keep OFF
                ])
                if not ok:
                    #log("[Top Capture] imencode failed")
                    continue

                top_image_queue.put(buf.tobytes())
                log(f"[Top Capture] Saved image")

            except Empty:
                time.sleep(0.02)
            except Exception as e:
                #log(f"[top Capture Error] {e}")
                time.sleep(0.05)
    finally:
        try:
            g.close()
        except Exception:
            pass

# Stage 2b: Expire Process (set TEST_DATE) -> queue to Label
def top_process(box_q_B2E, box_q_E2L: mp.Queue, top_image_queue: mp.Queue, ready):
    _assert_file(TOP_MODEL_PATH, "TOP_MODEL_PATH")
    model = YOLO(TOP_MODEL_PATH)
    prepare_yolo_model(model, DEVICE)
    warmup_yolo_for_detect(model, DEVICE, imgsz=480)  # fused, FP16 on CUDA, imgsz=480
    ready.wait()

    while True:
        # Receive JPEG bytes, decode to ndarray
        data = top_image_queue.get()  # blocking
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            #log("[top Process] imdecode failed")
            continue

        log("[top Process] Processing image...")
        try:
            results = model(img, conf=0.5, device=DEVICE, imgsz=480, verbose=False)[0]
        except Exception as e:
            log(f"[top YOLO Error] {e}")
            continue
        current_box = box_q_B2E.get()

        has_cap = False
        has_expire = False
        for b in results.boxes:
            cls = int(b.cls[0])
            cls_name = model.names[cls].lower()
            if cls_name == "cap":
                has_cap = True
            elif cls_name == "expiredate":
                has_expire = True

        current_box.set_cap(has_cap)
        current_box.set_expire_date("expire date printed" if has_expire else "no expire date")
        log(f"Top detection - cap: {has_cap}, expire date: {has_expire}")

        box_q_E2L.put(current_box)
        log(f"[top Process] Box #{current_box.id} -> box_q_E2L")

# --- Size classification via horizontal trigger line ---
SIZE_LINE_Y = 330   # <-- calibrate this once from a few frames
SIZE_LINE_MARGIN = 6  # small hysteresis so tiny jitters don't flip result
def intersects_horizontal_line(y1: int, y2: int, line_y: int, margin: int = 0) -> bool:
    # y grows downward in images; rectangle intersects a horizontal line if the line
    # lies between top (y1) and bottom (y2) with an optional margin.
    return (y1 - margin) <= line_y <= (y2 + margin)

# Label Worker helpers
def initialize_label_model():
    """Load YOLO and detection helper."""
    _assert_file(MODEL_PATH, "MODEL_PATH")
    model = YOLO(MODEL_PATH)
    prepare_yolo_model(model, DEVICE)
    return model, Detect()

def load_batch_data(batch_name: Optional[str]):
    """Load Excel batch file."""
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

#still uses trigger
def assign_box(track_id, cx, box_q_E2L, assigned, seen_tracks, processed_ids, trigger_x=TRIGGER_X):
    """Assign a pending box to a detected track if it crosses trigger (one-shot)."""
    # Do not assign if this track_id already processed
    if track_id in processed_ids:
        return
    if (track_id not in seen_tracks) and (cx >= trigger_x):
        try:
            pending_box = box_q_E2L.get_nowait()
            assigned[track_id] = pending_box
            seen_tracks.add(track_id)
            #log(f"[LabelWorker] Assigned Box #{pending_box.id} -> track {track_id}")
        except Empty:
            pass

# -------------- EPISODE GAP (handle reused track ids safely) ---------------
REUSE_GAP_S = 0.9  # if a track id is unseen longer than this, treat as a new physical object

def process_detection(key, class_name, assigned, detect, info_q, processed_ids, size_hint=None):
    """
    Episode-aware finalize:
    - key = (track_id, episode)
    - ensures we don't process the same object more than once
    - allows reusing the same track id later for a new object (new episode)
    """
    if key in processed_ids:
        assigned.pop(key, None)
        return

    box = assigned.get(key)
    if box is None:
        return

    try:
        parts = detect.label_detections(class_name)  # [brand, flavor, capacity, (opt) type]
        if not parts or len(parts) < 3:
            box.set_status("rejected")
            box.set_reason("parse_error")
        else:
            brand, flavor, capacity = parts[:3]

            # --- keep your horizontal-line size override ---
            if size_hint:
                if size_hint == "1l":
                    capacity = "1 L"
                else:
                    capacity = "235 ml"

            box.set_brand(brand)
            box.set_flavor(flavor)
            box.set_capacity(capacity)
            if len(parts) >= 4:
                box.set_product_type(parts[3])

            # continue with your usual evaluation logic
            box.evaluate_box()

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
        assigned.pop(key, None)
        processed_ids.add(key)   # mark (track_id, episode) processed

# 
# Stage 3: Label (OpenCV webcam)

# --- label_worker: instant track on first detection + robust pairing ---
# --- label_worker: no TRIGGER_X, instant track on first detection, robust pairing ---
def label_worker(
    camera_index: int,
    box_q_E2L: MPQueue,
    frame_q: MPQueue,
    info_q: MPQueue,
    stop_event: MPEvent,
    batch_name: Optional[str] = None,
    ready = None
):

    # -------- tunables --------
    IMG_SZ = 480
    CONF_TRACK = 0.70      # detector threshold for label cam tracking
    MIN_DET_CONF = 0.40    # min confidence to accept a track on first sight
    WAIT_TIMEOUT_S = 8.0   # drop waiting track only if no boxes pending and very old
    ORPHAN_TIMEOUT_S = 6.0 # how long a box can wait before fallback pairing
    RECENT_TRACK_HORIZON = 3.0  # consider keys seen within this many seconds for fallback
    FORWARD_JOIN_MAX_S = 8.0    # max time to keep parked orphans
    MAX_CARRY_ORPHANS  = 3      # safety cap for parked orphans

    time.sleep(1.5)
    cap = try_open_camera(camera_index)
    if not cap:
        return

    model, detect = initialize_label_model()
    warmup_yolo_for_detect(model, DEVICE, imgsz=IMG_SZ)
    if ready:
        ready.wait()
    if batch_name:
        load_batch_data(batch_name)

    # --- episode-aware state ---
    assigned = {}                 # key -> Box ; key = (track_id, episode)
    processed_ids = set()         # set of keys we've finished
    device = DEVICE
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # Buffers/state
    pending_boxes = deque()       # deque of (box, t_arrived) from E2L
    waiting_tracks = {}           # key -> {'t0','class_name','size_hint'}
    recent_tracks = {}            # key -> {'last_seen','class_name','size_hint'}
    carry_orphans = deque()       # parked orphans to forward-join with next key

    # Per-track episode info
    track_info = {}               # tid -> {'episode': int, 'last_seen': float}
    seen_keys  = set()            # fired (accepted) keys to avoid same-object duplicates

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            cap, ok, frame = recover_camera(camera_index)
            if not ok:
                break

        now = time.time()

        # Drain completed boxes (non-blocking)
        try:
            while True:
                b = box_q_E2L.get_nowait()
                pending_boxes.append((b, now))
        except Empty:
            pass

        # (Optional UI) keep size line if you want size_hint logic
        cv2.line(frame, (0, SIZE_LINE_Y), (frame.shape[1], SIZE_LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, f"SIZE LINE y={SIZE_LINE_Y}", (10, max(20, SIZE_LINE_Y - 8)),
                    FONT, 0.55, (0, 255, 255), 1)

        # --- YOLO tracking on label frame ---
        try:
            results = model.track(source=frame, conf=CONF_TRACK, persist=True,
                                  device=device, imgsz=IMG_SZ, verbose=False)
        except Exception:
            results = model.track(source=frame, conf=CONF_TRACK, persist=True,
                                  device="cpu", imgsz=IMG_SZ, verbose=False)
            device = "cpu"

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            names = getattr(results[0], "names", None) or getattr(model, "names", {})

            for i in range(len(boxes)):
                try:
                    track_id = int(boxes.id[i].item())
                    class_id = int(boxes.cls[i].item())
                    class_name = names[class_id] if isinstance(names, dict) else str(class_id)
                    conf_score = float(boxes.conf[i].item()) if getattr(boxes, "conf", None) is not None else 1.0
                    log(f"[LabelWorker] Detected track {track_id} '{class_name}' conf={conf_score:.2f}")

                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                    # (Optional UI)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, max(0, y1 - 8)),
                                FONT, 0.55, (0, 0, 255), 1)

                    # Size-hint from the horizontal line (kept for your capacity override)
                    hits_size_line = intersects_horizontal_line(y1, y2, SIZE_LINE_Y, margin=SIZE_LINE_MARGIN)
                    size_hint = "1l" if hits_size_line else "235ml"
                    cv2.putText(frame, f"size_hint:{size_hint}", (x1, max(0, y1 - 24)),
                                FONT, 0.55, (0, 255, 255), 1)

                    # --- EPISODE UPDATE for reused track ids ---
                    last = track_info.get(track_id)
                    if last is None:
                        track_info[track_id] = {'episode': 0, 'last_seen': now}
                    else:
                        if (now - last['last_seen']) > REUSE_GAP_S:
                            last['episode'] += 1  # new physical object under same track id
                        last['last_seen'] = now
                    key = (track_id, track_info[track_id]['episode'])

                    # Update recent track cache (for orphan fallback)
                    recent_tracks[key] = {
                        'last_seen': now,
                        'class_name': class_name,
                        'size_hint': size_hint
                    }

                    # === INSTANT acceptance on first detection (no TRIGGER_X) ===
                    if (key not in seen_keys) and (conf_score >= MIN_DET_CONF):
                        seen_keys.add(key)
                        if carry_orphans:
                            # Forward-join: pair oldest parked orphan with THIS newly accepted key
                            box, t_arrived = carry_orphans.popleft()
                            assigned[key] = box
                            try:
                                process_detection(key, class_name, assigned, detect, info_q, processed_ids,
                                                  size_hint=size_hint)
                                log(f"[LabelWorker] Forward-joined carry orphan box id={box.id} with key {key}")
                            except Exception as e:
                                log(f"[LabelWorker] Forward-join processing error: {e}")
                            finally:
                                recent_tracks.pop(key, None)  # don't reuse this key in fallback
                        else:
                            # Normal path: put into waiting queue
                            waiting_tracks[key] = {'t0': now, 'class_name': class_name, 'size_hint': size_hint}
                            log(f"[LabelWorker] INSTANT key {key} accepted (tid={track_id}, conf={conf_score:.2f}) "
                                f"(waiters={len(waiting_tracks)}, boxes={len(pending_boxes)})")
                    else:
                        # keep the latest label info while we still see it
                        if key in waiting_tracks:
                            waiting_tracks[key]['class_name'] = class_name
                            waiting_tracks[key]['size_hint']  = size_hint

                except Exception:
                    continue

        # Drop stale waiting tracks ONLY when no boxes are pending
        for k in list(waiting_tracks):
            age = now - waiting_tracks[k]['t0']
            if not pending_boxes and age > WAIT_TIMEOUT_S:
                log(f"[LabelWorker] Dropping stale waiting key {k} after {age:.2f}s (no boxes pending)")
                waiting_tracks.pop(k, None)

        # Primary pairing: earliest waiting key ↔ oldest completed box
        while pending_boxes and waiting_tracks:
            key_oldest = min(waiting_tracks, key=lambda k: waiting_tracks[k]['t0'])
            meta = waiting_tracks.pop(key_oldest)
            box, t_arrived = pending_boxes.popleft()

            assigned[key_oldest] = box
            try:
                process_detection(key_oldest, meta['class_name'], assigned, detect, info_q, processed_ids,
                                  size_hint=meta['size_hint'])
                log(f"[LabelWorker] Paired key {key_oldest} with box id={box.id} "
                    f"(waiters={len(waiting_tracks)}, boxes={len(pending_boxes)})")
                recent_tracks.pop(key_oldest, None)  # don't reuse this key in fallback
            except Exception as e:
                log(f"[LabelWorker] Processing error: {e}")

        # Orphan fallback: try eligible recent key; else park for forward-join
        if pending_boxes:
            box, t_arrived = pending_boxes[0]
            wait = now - t_arrived
            if wait > ORPHAN_TIMEOUT_S:
                candidates = [
                    k for k, v in recent_tracks.items()
                    if k not in processed_ids
                    and k not in assigned
                    and (now - v['last_seen']) <= RECENT_TRACK_HORIZON
                ]

                if candidates:
                    key_recent = max(candidates, key=lambda k: recent_tracks[k]['last_seen'])
                    tr = recent_tracks[key_recent]
                    meta = {'class_name': tr['class_name'], 'size_hint': tr['size_hint']}
                    pending_boxes.popleft()
                    assigned[key_recent] = box
                    try:
                        process_detection(key_recent, meta['class_name'], assigned, detect, info_q, processed_ids,
                                          size_hint=meta['size_hint'])
                        log(f"[LabelWorker] Fallback-paired orphan box id={box.id} with recent key {key_recent}")
                    except Exception as e:
                        log(f"[LabelWorker] Fallback processing error: {e}")
                    finally:
                        recent_tracks.pop(key_recent, None)
                else:
                    # No eligible key now -> park to forward-join with the next accepted key
                    if len(carry_orphans) < MAX_CARRY_ORPHANS:
                        carry_orphans.append(pending_boxes.popleft())
                        log(f"[LabelWorker] Parked orphan box id={box.id} for forward-join "
                            f"(carry={len(carry_orphans)})")
                    else:
                        log(f"[LabelWorker] Carry full; dropping orphan box id={box.id}")
                        pending_boxes.popleft()

        # Expire very old parked orphans to avoid indefinite waiting
        while carry_orphans and (now - carry_orphans[0][1]) > FORWARD_JOIN_MAX_S:
            old_box, t0 = carry_orphans.popleft()
            log(f"[LabelWorker] Dropping expired parked orphan box id={getattr(old_box,'id','?')}")

        # (Optional UI) preview
        try:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                if frame_q.full():
                    _ = frame_q.get_nowait()
                frame_q.put_nowait(buf.tobytes())
        except Exception:
            pass

    cap.release()




# Pipeline start/stop helpers for the UI

Pipeline = namedtuple("Pipeline", ["procs", "serial_thread", "frame_q", "info_q", "stop_event"])

def start_pipeline(label_cam_index: int = LABEL_CAM_INDEX,
                   frame_q: Optional[MPQueue] = None,
                   info_q: Optional[MPQueue] = None,
                   batch_name: Optional[str] = None):

    ready = Barrier(3)

    barcode_event_queue = mp.Queue()
    top_event_queue = mp.Queue()

    barcode_image_queue = mp.Queue()
    top_image_queue = mp.Queue()

    box_q_B2E = mp.Queue()
    box_q_E2L = mp.Queue()

    # UI streams
    frame_q = frame_q or mp.Queue(maxsize=2)
    info_q = info_q or mp.Queue(maxsize=64)
    stop_event = mp.Event()

    # Serial thread
    serial_thread = start_serial_listener(barcode_event_queue, top_event_queue)

    # Discover Allied Vision cameras
    av_ids = SoftTriggerGrabber.discover_av_cameras()
    if not av_ids:
        log("[Comm] No Allied Vision cameras found! Barcode/Expire capture will not start.")

    barcode_cam_id = av_ids[0] if len(av_ids) >= 1 else None
    top_cam_id  = av_ids[1] if len(av_ids) >= 2 else None

    procs = []

    # Barcode capture+process
    if barcode_cam_id:
        procs.append(mp.Process(target=barcode_capture_process,
                                args=(barcode_event_queue, barcode_image_queue, barcode_cam_id),
                                daemon=True))
        procs.append(mp.Process(target=barcode_process,
                                args=(barcode_image_queue, box_q_B2E, ready),
                                daemon=True))
    else:
        log("[Comm] Skipping barcode capture/process (no AV camera).")

    # Expiry capture + processing
    procs.append(mp.Process(target=top_capture_process,
                            args=(top_event_queue, top_image_queue, top_cam_id),
                            daemon=True))
    procs.append(mp.Process(target=top_process,
                            args=(box_q_B2E, box_q_E2L, top_image_queue, ready),
                            daemon=True))

    # Label worker
    procs.append(mp.Process(
        target=label_worker,
        args=(label_cam_index, box_q_E2L, frame_q, info_q, stop_event, batch_name, ready),
        daemon=True))

    for p in procs:
        p.start()

    #log("[Comm] Pipeline started (AV-triggered capture + JPEG queues).")
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