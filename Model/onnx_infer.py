# Model/onnx_infer.py
import os
import cv2
import numpy as np

def _ort():
    # Lazy import so module import won't fail if ORT isn't installed
    import onnxruntime as ort
    return ort

def new_dml_session(onnx_path: str):
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    ort = _ort()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    # Sanity: ensure DML is present when expected
    provs = sess.get_providers()
    print("ORT providers:", provs)  # you can comment this out after first run
    return sess

def _letterbox(im, new_shape=(640, 640), color=114):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
    pad_y, pad_x = (new_shape[0] - nh) // 2, (new_shape[1] - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, pad_x, pad_y

def run_yolo_onnx(sess, frame_bgr, conf_thr=0.25, imgsz=640):
    """
    Returns list of dicts: [{xyxy:(x1,y1,x2,y2), conf:float, cls:int}, ...].
    Assumes Ultralytics export with nms=True (output shape [N,6] or [1,N,6]).
    """
    im, r, pad_x, pad_y = _letterbox(frame_bgr, (imgsz, imgsz))
    im = im[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    im = np.transpose(im, (2, 0, 1))[None]  # NCHW

    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: im})

    dets = []
    arr = None
    for o in outs:
        a = np.array(o)
        if a.ndim == 2 and a.shape[1] == 6:
            arr = a
            break
        if a.ndim == 3 and a.shape[-1] == 6:
            arr = a.reshape(-1, 6)
            break

    if arr is None:
        # Fallback if model exported without NMS (implement your own NMS if needed)
        return dets

    for x1, y1, x2, y2, conf, cls in arr:
        if conf < conf_thr:
            continue
        # undo letterbox
        x1 = (x1 - pad_x) / r
        x2 = (x2 - pad_x) / r
        y1 = (y1 - pad_y) / r
        y2 = (y2 - pad_y) / r
        dets.append({
            "xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "conf": float(conf),
            "cls": int(cls)
        })
    return dets
