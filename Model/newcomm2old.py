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
        next_box_id += 1                    

        if len(results.boxes) == 0:
            new_box.set_barcode("inverted")
         
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

def process_detection(class_name,detect, info_q,box,size_hint=None):
   

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
                    capacity = "1L"
                else:
                    capacity = "235ml"

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



import cv2
import time
from collections import deque
from multiprocessing import Process, Event as MPEvent, Queue as MPQueue
from queue import Empty, Full

def _enqueue(q: MPQueue, item, drop_if_full=True):
    if q is None:
        return False
    try:
        q.put_nowait(item)
        return True
    except Full:
        if not drop_if_full:
            q.put(item)  # block if you prefer
        return False
    except Exception:
        return False

def label_capture_process(
    camera_index: int,
    event_queue: MPQueue,     # emits e.g. ("triggered", arduino_us, pc_time_s) or ("triggered", pc_time_s)
    frame_q: MPQueue,         # for UI display (newest frames)
    label_image_queue: MPQueue,       # frames to run detection when sensor triggers
    stop_event: MPEvent,
    ring_size: int = 10,      # number of recent frames kept (with timestamps)
    display_stride: int = 1,  # push every Nth frame to frame_q (reduce UI load)
    backend=cv2.CAP_DSHOW,    # helpful on Windows
):
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        # fallback
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[Capture] ERROR: Could not open camera index {camera_index}")
            return


    # ring buffer: (pc_time_s, frame)
    ring = deque(maxlen=ring_size)

    frame_idx = 0
    last_push_ui = 0

    print("[Capture] started")
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                # brief backoff and continue
                time.sleep(0.01)
                continue

            now = time.time()
            frame_idx += 1

            # keep in ring (copy only if you expect downstream mutation)
            ring.append((now, frame.copy()))

            # Throttle UI queue (every 'display_stride' frames)
            if display_stride <= 1 or (frame_idx % display_stride == 0):
               ok, ui_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                _enqueue(frame_q, ui_buf.tobytes(), drop_if_full=True)
                last_push_ui = now

            # Drain ALL pending events this loop so we don't fall behind
            while True:
                try:
                    ev = event_queue.get_nowait()
                except Empty:
                    break

                if not ev:
                    continue

                # Accept a few formats:
                # ("triggered", pc_time_s)
                # ("triggered", sensor_id, arduino_us, pc_time_s)
                # ("triggered", sensor_id, pc_time_s)
                # Normalize to (etype="triggered", pc_ts=float)
                etype = ev[0] if isinstance(ev, (list, tuple)) and len(ev) > 0 else None
                if etype != "triggered":
                    # Ignore other event types
                    continue

                # Try to parse a pc_time_s from event; if not present, use 'now'
                pc_ts = None
                if isinstance(ev, (list, tuple)):
                    # search any float-like in event tuple
                    for x in ev:
                        if isinstance(x, (int, float)) and x > 1000000000:  # rough heuristic for epoch seconds
                            pc_ts = float(x)
                            break
                if pc_ts is None:
                    pc_ts = now

                # Choose a frame from ring.
                # Strategy 1: latest
                chosen = ring[-1] if len(ring) else None

                # Strategy 2 (optional): nearest by timestamp
                # if len(ring):
                #     chosen = min(ring, key=lambda p: abs(p[0] - pc_ts))

                if chosen is None:
                    # If ring is empty (startup), just use current frame
                    chosen = (now, frame)

                chosen_frame = chosen[1] if chosen is not None else frame
                ok, trig_buf = cv2.imencode(".jpg", chosen_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    _enqueue(label_image_queue, trig_buf.tobytes(), drop_if_full=True)

            # Small sleep to keep CPU reasonable; tune as needed
            # If you need max FPS, set to 0 or a very small value
            # time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[Capture] stopped")




def label_process(box_q_E2L: mp.Queue, label_image_queue: mp.Queue,ready,batch_name: Optional[str],frame_q:mp.Queue,info_q:mp.Queue, stop_event: MPEvent):
    """Process label images, match with Box, and output results."""
    model, detect = initialize_label_model()
    warmup_yolo_for_detect(model, DEVICE, imgsz=640)
    if batch_name:
        load_batch_data(batch_name)
    ready.wait()

    while True:
        jpg_bytes = label_image_queue.get()  # blocking
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            # handle decode failure
            continue
   # blocking
        tag, event_id_box, current_box = box_q_E2L.get()   # blocking


        try:
            results = model.track(frame, conf=0.8, device=DEVICE)[0]
        except Exception:
            continue

        for b in results.boxes:
            cls = int(b.cls[0])
            cls_name = model.names[cls].lower()
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            size_hint=intersects_horizontal_line(y1, y2, SIZE_LINE_Y, SIZE_LINE_MARGIN)
            process_detection(cls_name,detect,info_q,current_box,size_hint)

       


# Pipeline start/stop helpers for the UI

Pipeline = namedtuple("Pipeline", ["procs", "serial_thread", "frame_q", "info_q", "stop_event"])

def start_pipeline(label_cam_index: int = LABEL_CAM_INDEX,
                   frame_q: Optional[MPQueue] = None,
                   info_q: Optional[MPQueue] = None,
                   batch_name: Optional[str] = None):
    
    ready = Barrier(3)

    barcode_event_queue = mp.Queue()
    top_event_queue=mp.Queue()
    label_event_queue=mp.Queue()

    barcode_image_queue = mp.Queue()
    top_image_queue=mp.Queue()
    label_image_queue=mp.Queue()    

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
        target=label_capture_process,
        args=(label_cam_index,          # camera_index
            label_event_queue,        # event_queue
            frame_q,                  # frame_q (for UI)
            label_image_queue,        # label_image_queue (triggered)
            stop_event),              # stop_event
        daemon=True))

    # --- Label process (consume triggered frames + boxes -> detect -> save info)
    procs.append(mp.Process(
        target=label_process,
        args=(box_q_E2L,                # box_q_E2L
            label_image_queue,        # label_image_queue
            ready,                    # ready (barrier)
            batch_name,               # batch_name
            frame_q,                  # frame_q (optional for overlays/logs)
            info_q,                   # info_q (UI/status)
            stop_event),              # stop_event
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