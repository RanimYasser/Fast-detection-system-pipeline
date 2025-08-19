import serial 
import time
import threading
import multiprocessing as mp
from Model.utils import log
SERIAL_PORT = "COM4"
BAUD_RATE = 115200

def serial_listener_thread(barcode_event_queue: mp.Queue):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        log(f"[Serial] Listening on {SERIAL_PORT} ...")
        while True:
            try:
                line = ser.readline().decode(errors="ignore").strip().lower()
                # Accept a few spellings and normalize to "pressed"
                if line == "switch pressed":
                    t = time.time()
                    log("[Serial] Trigger received.")
                    barcode_event_queue.put(("switch pressed", t))
            except Exception as e:
                log(f"[Serial Error] {e}")
                time.sleep(0.05)
    except serial.SerialException as e:
        log(f"[Serial Error] Could not open serial port: {e}")


def start_serial_listener(barcode_event_queue: mp.Queue):
    """Start the serial listener thread."""
    serial_thread = threading.Thread(target=serial_listener_thread, args=(barcode_event_queue,), daemon=True)
    serial_thread.start()
    return serial_thread
