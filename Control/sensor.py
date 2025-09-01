import serial, time, threading, multiprocessing as mp
from Model.utils import log

SERIAL_PORT = "COM6"
BAUD_RATE = 115200

def serial_listener_thread(barcode_event_queue: mp.Queue,
                           top_event_queue: mp.Queue):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        log(f"[Serial] Listening on {SERIAL_PORT} ...")
        while True:
            try:
                raw = ser.readline()
                if not raw:
                    time.sleep(0.01)
                    continue

                line = raw.decode(errors="ignore").strip()
                low  = line.lower()

                # --- Back-compat with your old sketch ---
                if low == "switch pressed":
                    t = time.time()
                    log("[Serial] Trigger D2 (compat) received.")
                    barcode_event_queue.put(("switch pressed", t))
                    continue

                # --- New format: "D2: PRESSED", "D3: PRESSED" ---
                # Accept minor variations in spacing/case
                if ":" in low:
                    left, right = [s.strip() for s in low.split(":", 1)]
                    if right == "pressed":
                        t = time.time()
                        if left == "d2":
                            log("[Serial] Trigger D2 received.")
                            barcode_event_queue.put(("switch pressed", t))
                        elif left == "d3":
                            log("[Serial] Trigger D3 received.")
                            top_event_queue.put(("switch pressed", t))
            except Exception as e:
                log(f"[Serial Error] {e}")
                time.sleep(0.05)
    except serial.SerialException as e:
        log(f"[Serial Error] Could not open serial port: {e}")

def start_serial_listener(barcode_event_queue: mp.Queue,
                          top_event_queue: mp.Queue):
    """Start the serial listener thread (daemon)."""
    serial_thread = threading.Thread(
        target=serial_listener_thread,
        args=(barcode_event_queue, top_event_queue),
        daemon=True
    )
    serial_thread.start()
    return serial_thread