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
from multiprocessing import Barrier

import torch

# --- add these two early ---
import torch
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Control.ReadExcel import readExcel, findProduct, matchBarcode, product_from_parts
from Model.utils import log, save_box_info, _assert_file, try_open_camera
from Control.sensor import start_serial_listener
from Control.camera_manger2 import SoftTriggerGrabber

# Config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_label.pt"
BARCODE_MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_barcode.pt"
TOP_MODEL_PATH = r"C:\Users\RC-co\Desktop\Fast-detection\Yolo-models\best_top.pt"
TEST_DATE = True
LABEL_CAM_INDEX = 0
#TRIGGER_X = 200
BARCODE_SAVE_DIR = os.path.join("captures", "barcode")
os.makedirs(BARCODE_SAVE_DIR, exist_ok=True)
TOP_SAVE_DIR = os.path.join("captures", "top")
os.makedirs(TOP_SAVE_DIR, exist_ok=True)

# Project imports (relative safe import)
try:
    from Model.box import Box
    from Model.detection import Detect
except ModuleNotFoundError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT_DIR)
    from Model.box import Box
    from Model.detection import Detect

# Device picker (CUDA / DirectML / CPU)

def warmup_yolo_for_detect(model, device="cpu", imgsz=640):
    import numpy as np, cv2
    dummy = np.zeros((imgsz, imgsz, 3), np.uint8)
    # One detect forward
    _ = model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    # One more to stabilize memory allocator
    _ = model.predict(dummy, imgsz=imgsz, device=device, verbose=False)


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
    #log("[Barcode Capture] Ready. Waiting for press...")

    try:
        while True:
            try:
                event, event_id = event_queue.get(timeout=1)
                if event != "triggered":
                    continue

                # Trigger a new exposure and get the frame
                img = g.fire(timeout=0.3)
                if img is None:
                    #log("[Barcode Capture] Trigger timeout (no frame).")
                    continue

                # Save with timestamp and cam id
                # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # fname = f"{ts}{cam_id.replace(':','').replace('/','_')}.jpg"
                # fpath = os.path.join(BARCODE_SAVE_DIR, fname)
                # try:
                #     ok = cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                #     if ok:
                #         log(f"[Barcode Capture] Saved {fname}")
                #     else:
                #         log(f"[Barcode Capture] Failed to save {fname}")
                # except Exception as e:
                #     log(f"[Barcode Capture] Save error: {e}")

                # Put JPEG bytes on queue (Windows-picklable)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    #log("[Barcode Capture] imencode failed")
                    continue
                barcode_image_queue.put((event_id, buf.tobytes()))


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
    warmup_yolo_for_detect(model, DEVICE, imgsz=640)
    ready.wait()
    detect = Detect()
    next_box_id = 0

    while True:
        # BEFORE (you had two gets; remove the first)
        # data = barcode_image_queue.get()
        # AFTER
        event_id, data = barcode_image_queue.get()  # blocking

        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue

        try:
            results = model(img, conf=0.7, device=DEVICE)[0]
        except Exception:
            continue

        new_box = Box()
        new_box.set_id(next_box_id)
        next_box_id += 1                    # ✅ (remove your second increment)

        if len(results.boxes) == 0:
            new_box.set_barcode(None)
        else:
            decoded_any = False
            for b in results.boxes:
                cls = int(b.cls[0])
                cls_name = model.names[cls]
                if cls_name.lower() == "barcode":
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    roi = img[y1:y2, x1:x2]
                    try:
                        decoded = detect.barcode_detection(roi)
                    except Exception:
                        decoded = None
                    if decoded:
                        new_box.set_barcode(decoded)
                        decoded_any = True
                        break
            if not decoded_any:
                new_box.set_barcode(None)

        box_q_B2E.put((event_id, new_box))  # ✅ carry the id forward


def initialize_top_model():
    _assert_file(TOP_MODEL_PATH, "TOP_MODEL_PATH")
    return YOLO(TOP_MODEL_PATH)


def top_capture_process(event_queue: mp.Queue,top_image_queue:mp.Queue ,cam_id: str):
    #log(f"[top Capture] Using AV camera id={cam_id}")
    g = SoftTriggerGrabber(cam_id, pixel_format='BGR8')
    #log(f"[top Capture] Saving to: {TOP_SAVE_DIR}")
    #log("[top Capture] Ready. Waiting for press...")

    try:
        while True:
            try:
                event, event_id = event_queue.get(timeout=1)
                if event != "triggered":
                    continue

                # Trigger a new exposure and get the frame
                img = g.fire(timeout=0.3)
                if img is None:
                    log("[TOP Capture] Trigger timeout (no frame).")
                    continue

                #Save with timestamp and cam id
                # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                # fname = f"{ts}{cam_id.replace(':','').replace('/','_')}.jpg"
                # fpath = os.path.join(TOP_SAVE_DIR, fname)
                # try:
                #     ok = cv2.imwrite(fpath, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                #     if ok:
                #         log(f"[Top Capture] Saved {fname}")
                #     else:
                #         log(f"[Top Capture] Failed to save {fname}")
                # except Exception as e:
                #     log(f"[Top Capture] Save error: {e}")

                # Put JPEG bytes on queue (Windows-picklable)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    #log("[Top Capture] imencode failed")
                    continue
                top_image_queue.put((event_id, buf.tobytes()))

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
    warmup_yolo_for_detect(model, DEVICE, imgsz=640)
    ready.wait()

    while True:
        # Get both queues FIRST
        event_id_img, data = top_image_queue.get()   # blocking
        event_id_box, current_box = box_q_B2E.get()  # blocking

        # Optional sanity check:
        # if event_id_img != event_id_box:
        #     log(f"[top_process] ID mismatch: img={event_id_img} box={event_id_box}")

        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            # keep pipeline flowing even if decode fails
            box_q_E2L.put(("box", event_id_box, current_box))
            continue

        try:
            results = model(img, conf=0.5, device=DEVICE)[0]
        except Exception:
            box_q_E2L.put(("box", event_id_box, current_box))
            continue

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
        box_q_E2L.put(("box", event_id_box, current_box))   # what the matcher expects

        #log(f"[top Process] Box #{current_box.id} -> box_q_E2L")
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
    return YOLO(MODEL_PATH), Detect()

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
def assign_box(track_id, cx, box_q_E2L, assigned, seen_tracks, processed_ids):
    """Assign a pending box to a detected track if it crosses trigger (one-shot)."""
    # Do not assign if this track_id already processed
    if track_id in processed_ids:
        return
    if track_id not in seen_tracks:
        try:
            pending_box = box_q_E2L.get_nowait()
            assigned[track_id] = pending_box
            seen_tracks.add(track_id)
            #log(f"[LabelWorker] Assigned Box #{pending_box.id} -> track {track_id}")
        except Empty:
            pass

def process_detection(track_id, class_name, assigned, detect, info_q, processed_ids, size_hint=None):
    if track_id in processed_ids:
        assigned.pop(track_id, None)
        return

    box = assigned.get(track_id)
    if box is None:
        return

    try:
        parts = detect.label_detections(class_name)  # [brand, flavor, capacity, (opt) type]
        if not parts or len(parts) < 3:
            box.set_status("rejected")
            box.set_reason("parse_error")
        else:
            brand, flavor, capacity = parts[:3]

            # --- NEW: apply horizontal-line size override ---
            if size_hint:
                # normalize then slam to the enforced capacity
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
        assigned.pop(track_id, None)
        processed_ids.add(track_id)

# 
# Stage 3: Label (OpenCV webcam)
# 
from collections import deque

# timeouts so nothing blocks forever
BOX_WAIT_TIMEOUT_S    = 1.2
LABEL_WAIT_TIMEOUT_S  = 1.2

def label_box_matcher_proc(
    box_q_E2L: mp.Queue,
    labels_queue: mp.Queue,
    info_q: mp.Queue,
    stop_event: mp.Event,
    batch_name: Optional[str] = None,     # add this
):
    # Ensure Excel DF is loaded in THIS process
    if batch_name:
        try:
            readExcel(f"{batch_name}.xlsx")
        except Exception as e:
            log(f"[Matcher] readExcel error: {e}")

    detect = Detect()  # real parser

    pending_boxes  = {}
    pending_labels = {}
    def _now(): return time.perf_counter()

    while not stop_event.is_set():
        # pump boxes
        while True:
            try:
                item = box_q_E2L.get_nowait()
                if isinstance(item, tuple) and len(item) >= 3 and item[0] == "box":
                    _, event_id, box_obj = item
                    pending_boxes[event_id] = {"t_recv": _now(), "box": box_obj}
            except Empty:
                break
            except Exception:
                break

        # pump labels
        while True:
            try:
                event_id, cls_name, conf, t_trig, meta = labels_queue.get_nowait()
                pending_labels[event_id] = {
                    "t_recv": _now(),
                    "label": {"class": cls_name, "conf": conf, "t_trigger": t_trig, "meta": meta}
                }
            except Empty:
                break
            except Exception:
                break

        # match
        for eid in list(pending_boxes.keys() & pending_labels.keys()):
            box_obj = pending_boxes[eid]["box"]
            lab     = pending_labels[eid]["label"]

            # optional size hint from label bbox vs SIZE_LINE_Y
            size_hint = None
            try:
                bbox = lab["meta"].get("bbox")
                if bbox:
                    y1, y2 = int(bbox[1]), int(bbox[3])
                    hits = intersects_horizontal_line(y1, y2, SIZE_LINE_Y, margin=SIZE_LINE_MARGIN)
                    size_hint = "1l" if hits else "235ml"
            except Exception:
                pass

            try:
                assigned = {eid: box_obj}
                process_detection(eid, lab["class"], assigned, detect, info_q,
                                  processed_ids=set(), size_hint=size_hint)
            except Exception as e:
                log(f"[Matcher] finalize error: {e}")

            pending_boxes.pop(eid, None)
            pending_labels.pop(eid, None)

        # expire stragglers
        now = _now()
        for eid in [k for k,v in pending_boxes.items() if (now - v["t_recv"]) >= BOX_WAIT_TIMEOUT_S]:
            try:
                assigned = {eid: pending_boxes[eid]["box"]}
                process_detection(eid, "Undefined", assigned, detect, info_q, processed_ids=set(), size_hint=None)
            except Exception as e:
                log(f"[Matcher] expire box error: {e}")
            pending_boxes.pop(eid, None)

        for eid in [k for k,v in pending_labels.items() if (now - v["t_recv"]) >= LABEL_WAIT_TIMEOUT_S]:
            # drop unmatched labels (no box)
            pending_labels.pop(eid, None)

        time.sleep(0.002)


import time, multiprocessing as mp
import cv2
from queue import Empty

# ---- knobs (tune to taste) ----
TRIGGER_X             = 360      # virtual vertical line (px)
DIR_RIGHT_TO_LEFT     = True     # your conveyor direction

NEAR_LINE_ONLY        = False     # <- disable any near-line gating
SNAPSHOT_MAX_AGE_MS   = 600       # allow a bit more slack than 400
DET_CONF              = 0.35      # make detections easier to get
SLEEP_BETWEEN_FRAMES  = 0.003
def _pick_current_label(results, model_names):
    """
    Returns {'class': str, 'conf': float, 'bbox': (x1,y1,x2,y2)} or None.
    Picks highest-confidence detection in the frame. No spatial gating.
    """
    boxes = None
    try:
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
    except Exception:
        return None
    if boxes is None:
        return None

    names = model_names or {}
    try:
        n = len(boxes)
    except Exception:
        n = 0
    if n == 0:
        return None

    best = None
    best_conf = -1.0
    for i in range(n):
        try:
            cls  = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1,y1,x2,y2 = map(int, boxes.xyxy[i].tolist())
            if conf > best_conf:
                best_conf = conf
                best = {"class": names.get(cls, str(cls)),
                        "conf": conf,
                        "bbox": (x1, y1, x2, y2)}
        except Exception:
            continue
    return best

def label_detector_sensor_proc(
    camera_index: int,
    event_queue: mp.Queue,      # ('triggered', event_id, ...) from your sensor thread
    labels_queue: mp.Queue,     # (event_id, class, conf, t_trigger_pc, meta)
    frame_q: mp.Queue,          # optional: JPEG bytes for UI
    stop_event: mp.Event,
    *,
    snapshot_max_age_ms: int = 600,   # how far back we accept a detection for a trigger
    det_conf: float = 0.35,           # YOLO confidence (lower to ensure we get boxes)
    burst_ring_size: int = 12,        # how many recent detections to keep
    sleep_between_frames: float = 0.003
):
    """
    Sensor-only fusion:
      - Continuously runs YOLO tracking on the label camera.
      - Maintains a small ring buffer of (timestamp, best_detection).
      - When a sensor trigger arrives, picks the most recent detection within `snapshot_max_age_ms`
        and pushes (event_id, class, conf, t_trigger, meta) to labels_queue.
      - No spatial gating / trigger line involved.
    """
    from collections import deque

    # ---- open camera ----
    cap = try_open_camera(camera_index)
    if not cap:
        return

    # ---- model init ----
    try:
        model, _ = initialize_label_model()
    except Exception as e:
        log(f"[LabelDet] init error: {e}")
        try:
            cap.release()
        except Exception:
            pass
        return

    try:
        warmup_yolo_for_detect(model, DEVICE, imgsz=640)
    except Exception:
        pass
    model_names = getattr(model, "names", {}) or {}

    # ---- small buffer of recent detections ----
    snapshots = deque(maxlen=burst_ring_size)   # each item: (t_perf, {'class','conf','bbox'})

    def _pick_top1(results):
        """
        Return {'class': str, 'conf': float, 'bbox': (x1,y1,x2,y2)} or None.
        Chooses the highest-confidence detection in the current frame.
        """
        try:
            if not results or results[0].boxes is None:
                return None
        except Exception:
            return None

        boxes = results[0].boxes
        try:
            n = len(boxes)
        except Exception:
            n = 0
        if n == 0:
            return None

        best = None
        best_conf = -1.0
        for i in range(n):
            try:
                cls  = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "class": model_names.get(cls, str(cls)),
                        "conf": conf,
                        "bbox": (x1, y1, x2, y2),
                    }
            except Exception:
                continue
        return best

    # ---- main loop ----
    while not stop_event.is_set():
        # 1) grab frame
        ok, frame = (False, None)
        try:
            ok, frame = cap.read()
        except Exception:
            ok = False

        if not ok or frame is None:
            try:
                cap, ok, frame = recover_camera(camera_index)
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.03)
                # still drain sensor queue to avoid buildup
                try:
                    while True:
                        _ = event_queue.get_nowait()
                except Empty:
                    pass
                except Exception:
                    pass
                continue

        # 2) run tracking on this frame
        try:
            results = model.track(
                source=frame,
                conf=det_conf,
                persist=True,
                device=DEVICE,
                verbose=False
            )
        except Exception as e:
            results = None
            log(f"[LabelDet] model.track error: {e}")

        # 3) take a snapshot (top-1 by confidence)
        det = _pick_top1(results)
        if det is not None:
            snapshots.append((time.perf_counter(), det))
            # debug (optional): log(f"[LabelDet] snap {det['class']} {det['conf']:.2f}")

        # 4) drain sensor triggers and emit labels (time-based only)
        while True:
            try:
                evt = event_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

            if isinstance(evt, tuple) and len(evt) >= 2 and evt[0] == "triggered":
                _, event_id = evt[:2]
                t_trigger = time.perf_counter()

                # find most recent snapshot within age window
                chosen = None
                for t_snap, d in reversed(snapshots):
                    if (t_trigger - t_snap) * 1000.0 <= snapshot_max_age_ms:
                        chosen = d
                        break

                if chosen is not None:
                    out_class = chosen["class"]
                    out_conf  = chosen["conf"]
                    meta      = {"bbox": chosen.get("bbox")}
                else:
                    out_class = "Undefined"
                    out_conf  = 0.0
                    meta      = {}

                try:
                    log(f"[LabelDet] Trigger eid={event_id} -> {out_class} ({out_conf:.2f})")
                    labels_queue.put_nowait((event_id, out_class, out_conf, t_trigger, meta))
                except Exception:
                    pass
                # debug (optional): log(f"[LABEL] eid={event_id} class={out_class} conf={out_conf:.2f}")

        # 5) optional: push preview frame as JPEG (non-blocking)
        try:
            ok_jpg, buf = cv2.imencode(".jpg", frame)
            if ok_jpg:
                if frame_q.full():
                    try:
                        _ = frame_q.get_nowait()
                    except Exception:
                        pass
                frame_q.put_nowait(buf.tobytes())
        except Exception:
            pass

        time.sleep(sleep_between_frames)

    # ---- cleanup ----
    try:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
    except NameError:
        pass


# Pipeline start/stop helpers for the UI

Pipeline = namedtuple("Pipeline", ["procs", "serial_thread", "frame_q", "info_q", "stop_event"])

def start_pipeline(label_cam_index: int = LABEL_CAM_INDEX,
                   frame_q: Optional[MPQueue] = None,
                   info_q: Optional[MPQueue] = None,
                   batch_name: Optional[str] = None):
    
    ready = Barrier(2)

    barcode_event_queue = mp.Queue()
    top_event_queue=mp.Queue()
    label_event_queue=mp.Queue()

    barcode_image_queue = mp.Queue()
    top_image_queue=mp.Queue()
    box_q_B2E = mp.Queue()
    box_q_E2L = mp.Queue()
    labels_queue = mp.Queue()

    # UI streams
    frame_q = frame_q or mp.Queue(maxsize=2)
    info_q = info_q or mp.Queue(maxsize=64)
    stop_event = mp.Event()
    

    # Serial thread
    serial_thread = start_serial_listener(barcode_event_queue, top_event_queue,label_event_queue)

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
                                args=(barcode_image_queue, box_q_B2E,ready),
                                daemon=True))
    else:
        log("[Comm] Skipping barcode capture/process (no AV camera).")

    # Expiry capture + processing
    procs.append(mp.Process(target=top_capture_process,
                            args=(top_event_queue, top_image_queue,top_cam_id   ),
                            daemon=True))
    procs.append(mp.Process(target=top_process,
                            args=(box_q_B2E, box_q_E2L,top_image_queue,ready),
                            daemon=True))

    # Label worker
    procs.append(mp.Process(
        target=label_detector_sensor_proc,
                    args=(label_cam_index, label_event_queue, labels_queue, frame_q, stop_event),
                    daemon=True))
    procs.append(mp.Process(target=label_box_matcher_proc,
                    args=(box_q_E2L, labels_queue, info_q, stop_event,batch_name),
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
    #log("[Comm] Pipeline stopped.")