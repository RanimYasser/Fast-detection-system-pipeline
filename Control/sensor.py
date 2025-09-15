# serial_timestamp_listener.py (legacy-free)
import serial, time, threading, multiprocessing as mp
from typing import Tuple, Optional
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
        pass  # drop if full

def _listen_loop(barcode_event_queue:mp.Queue,top_event_queue:mp.Queue,label_event_queue:mp.Queue, port: str, baud: int):
    """
    Reads frames: 'T' (0x54), sensor_id (0/1), <uint32 micros LE>
    Enqueues to q0 for id=0 (D2) and q1 for id=1 (D3).

    Queue item:
      ("triggered", sensor_id, arduino_us, pc_time_s, delta_ms)
    """
    top_event_id=0
    barcode_event_id=0
    # label_event_id=0
    ser = None
    while True:
        try:
            if ser is None:
                ser = serial.Serial(port, baudrate=baud, timeout=READ_TIMEOUT)
                time.sleep(BOOT_WAIT)
                ser.reset_input_buffer()

            b = ser.read(1)
            if not b:
                continue
            if b != b'T':
                continue  # resync

            id_b = ser.read(1)
            if len(id_b) != 1:
                continue  # short read, resync

            ts_b = ser.read(4)
            if len(ts_b) != 4:
                continue  # short read, resync

            sensor_id = id_b[0]  # 0 (D2) or 1 (D3)
            # t_arduino_us = int.from_bytes(ts_b, 'little', signed=False)
            # t_pc = time.time()
            # delta_ms = (t_pc * 1e6 - t_arduino_us) / 1000.0
           
            if sensor_id == 0:
                item = ("triggered", barcode_event_id)

                _enqueue(barcode_event_queue, item)
                barcode_event_id += 1
            elif sensor_id == 1:
                item = ("triggered", top_event_id)

                _enqueue(top_event_queue, item)
                top_event_id += 1

            # elif sensor_id == 2:
            #     item = ("triggered", label_event_id)
            #     log|(f"[SerialListener] Label event_id={label_event_id}")

            #     _enqueue(label_event_queue, item)
            #     label_event_id += 1
            # ignore any other IDs

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
        args=(barcode_event_queue, top_event_queue,label_event_queue, port, baud),
        daemon=True,
        name="SerialTimestampListener",
    )
    t.start()
    return t
