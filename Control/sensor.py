# serial_timestamp_listener.py (legacy-free, 3 sensors)
import serial, time, threading, multiprocessing as mp
from typing import Optional
from Model.utils import log

DEFAULT_PORT  = "COM7"
DEFAULT_BAUD  = 115200
READ_TIMEOUT  = 0.02   # seconds
BOOT_WAIT     = 1.0
REOPEN_DELAY  = 0.5

def _enqueue(q: Optional[mp.Queue], item) -> None:
    if q is None:
        return
    try:
        q.put_nowait(item)
    except Exception:
        # drop if full / consumer slow
        pass

def _listen_loop(
    barcode_event_queue: mp.Queue,
    top_event_queue: mp.Queue,
    label_event_queue: mp.Queue,
    port: str,
    baud: int,
):
    """
    Reads frames: 'T' (0x54), sensor_id (0/1/2), <uint32 micros LE>
    Routes:
      id=0 -> barcode_event_queue
      id=1 -> top_event_queue
      id=2 -> label_event_queue
    Enqueued item: ("triggered", local_event_id)
    """
    counters = {0: 0, 1: 0, 2: 0}
    ser = None
    while True:
        try:
            if ser is None:
                ser = serial.Serial(port, baudrate=baud, timeout=READ_TIMEOUT)
                time.sleep(BOOT_WAIT)
                ser.reset_input_buffer()

            b = ser.read(1)
            if not b or b != b'T':
                continue  # resync

            id_b = ser.read(1)
            if len(id_b) != 1:
                continue  # short read, resync

            ts_b = ser.read(4)
            if len(ts_b) != 4:
                continue  # short read, resync

            sensor_id = id_b[0]  # 0=D2(barcode), 1=D3(top), 2=LiDAR/TOF

            # If you need timestamps later, uncomment:
            # t_arduino_us = int.from_bytes(ts_b, 'little', signed=False)
            # t_pc = time.time()
            # delta_ms = (t_pc * 1e6 - t_arduino_us) / 1000.0

            if sensor_id == 0:
                _enqueue(barcode_event_queue, ("triggered", counters[0]))
                counters[0] += 1

            elif sensor_id == 1:
                _enqueue(top_event_queue, ("triggered", counters[1]))
                counters[1] += 1

            elif sensor_id == 2:
                log(f"[SerialListener] TOF/LiDAR event_id={counters[2]}")
                _enqueue(label_event_queue, ("triggered", counters[2]))
                counters[2] += 1

            else:
                # Unknown ID — ignore (or log if you want)
                # log(f"[SerialListener] Unknown sensor_id={sensor_id}")
                continue

        except serial.SerialException:
            try:
                if ser:
                    ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(REOPEN_DELAY)

        except Exception:
            time.sleep(REOPEN_DELAY)

def start_serial_listener(
    barcode_event_queue: mp.Queue,
    top_event_queue: mp.Queue,
    label_event_queue: mp.Queue,
    *,
    port: str = DEFAULT_PORT,
    baud: int = DEFAULT_BAUD,
) -> threading.Thread:
    t = threading.Thread(
        target=_listen_loop,
        args=(barcode_event_queue, top_event_queue, label_event_queue, port, baud),
        daemon=True,
        name="SerialTimestampListener",
    )
    t.start()
    return t
