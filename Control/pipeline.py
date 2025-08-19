# Control/pipeline.py

import multiprocessing as mp
import threading

from Model.communication import (
    serial_listener_thread, test_event_pumper,
    barcode_capture_process, expire_date_capture_process,
    barcode_process, expire_date_process, label_process,
    save_box_info
)

def start_pipeline(results_q: mp.Queue,
                   *,
                   label_cam_source=0,       # int (OpenCV index) or "DEV_..."
                   barcode_cam_id=None,      # "DEV_..." or None→autodetect [0]
                   expire_cam_id=None,       # "DEV_..." or None→autodetect [1]
                   label_from_expire=False   # keep False: label uses its own cam
                   ):
    """
    Returns (procs_list, frame_q)
      - frame_q: JPEG bytes for Qt CameraPreview (annotated label frames)
      - results_q: finalized box dicts (caller-provided)
    """
    # stage queues
    event_queue = mp.Queue()
    barcode_image_queue = mp.Queue()
    expire_date_image_queue = mp.Queue()
    shared_queue_1 = mp.Queue()  # barcode → expire
    shared_queue_2 = mp.Queue()  # expire  → label

    # GUI annotated frames from label stage
    frame_q = mp.Queue(maxsize=5)

    save_box_info({"event": "startup"}, track_id=-1, class_name="__startup__")

    # background threads (parent)
    threading.Thread(target=serial_listener_thread, args=(event_queue,), daemon=True).start()
    threading.Thread(target=test_event_pumper,     args=(event_queue,), daemon=True).start()

    # processes
    procs = []
    procs.append(mp.Process(target=barcode_capture_process,
                            args=(event_queue, barcode_image_queue, barcode_cam_id)))
    procs.append(mp.Process(target=expire_date_capture_process,
                            args=(event_queue, expire_date_image_queue, expire_cam_id, None)))
    procs.append(mp.Process(target=barcode_process,
                            args=(barcode_image_queue, shared_queue_1)))
    procs.append(mp.Process(target=expire_date_process,
                            args=(expire_date_image_queue, shared_queue_1, shared_queue_2)))
    procs.append(mp.Process(target=label_process,
                            args=(label_cam_source, shared_queue_2, frame_q, results_q)))

    for p in procs: p.start()
    return procs, frame_q


def stop_pipeline(procs):
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=2.0)
