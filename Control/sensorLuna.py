# reliable_serial_listener.py
import serial, time, threading, multiprocessing as mp, json
from Model.utils import log  # uses your existing logger
from datetime import datetime

# === Config ===
SERIAL_PORT = "COM7"
BAUD_RATE   = 115200
READ_CHUNK  = 256          # bytes per read
SER_TIMEOUT = 0.2          # small blocking timeout (prevents partial-line drops)
REOPEN_DELAY = 1.0         # seconds before trying to reopen after an error
BOOT_WAIT    = 2.0         # Mega/Nano auto-reset on open

# Optional: suppress rapid re-triggers (ms) if your Arduino fires multiple JSONs
DEBOUNCE_MS = 0            # set e.g. 40–80 if you see duplicate triggers

def _open_serial(port: str, baud: int) -> serial.Serial:
    """Open serial with sane defaults and give Arduino time to reboot."""
    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=SER_TIMEOUT,
        rtscts=False,
        dsrdtr=False,
        xonxoff=False,
        write_timeout=1.0,
    )
    time.sleep(BOOT_WAIT)       # let the board reboot after DTR toggle
    ser.reset_input_buffer()    # drop boot noise
    return ser

def _iter_json_lines(ser: serial.Serial):
    """
    Yield parsed JSON dicts from an async stream.
    Robust to partial reads and stray bytes. Returns None when no new bytes in this tick.
    """
    buf = b""
    while True:
        try:
            chunk = ser.read(READ_CHUNK)
        except serial.SerialException:
            # Let caller handle reconnect; break this generator
            break

        if not chunk:
            # no new bytes this tick
            yield None
            continue

        buf += chunk
        # split on newline, keep remainder in buf
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            s = line.decode('utf-8', 'ignore').strip()
            if not s:
                continue

            # find a {...} payload even if there’s leading/trailing noise
            i, j = s.find('{'), s.rfind('}')
            if i != -1 and j != -1 and j > i:
                payload = s[i:j+1]
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    # malformed line; skip
                    continue
            # else: no full JSON object in this line → ignore

def serial_listener_thread(barcode_event_queue: mp.Queue,
                           top_event_queue: mp.Queue,
                           port: str = SERIAL_PORT,
                           baud: int = BAUD_RATE):
    """
    Robust listener:
      - Small blocking reads (timeout ~0.2s) to avoid partial-line drops
      - Buffers fragments until newline
      - Reconnects on error/unplug
      - Optional de-bounce per sensor
    Routes:
      sensor "A" → top_event_queue
      sensor "B" → barcode_event_queue
    """
    last_fire_ts = {"A": 0.0, "B": 0.0}
    debounce_s = (DEBOUNCE_MS or 0) / 1000.0

    while True:
        ser = None
        try:
            ser = _open_serial(port, baud)
            log(f"[Serial] Listening on {port} @ {baud} …")

            hb_count = trig_count = 0
            t0 = time.time()

            for msg in _iter_json_lines(ser):
                if msg is None:
                    # idle tick — you can add link watchdogs here if desired
                    continue

                evt = str(msg.get("event") or "").lower()
                if evt == "hb":
                    hb_count += 1
                    # (Optionally) inspect distances during HB:
                    # d1 = msg.get("d1") or msg.get("d1_mm") or 0
                    # d2 = msg.get("d2") or msg.get("d2_mm") or 0
                    # log(f"[HB] d1={d1} d2={d2}")
                    pass

                elif evt == "lidar_trigger":
                    sensor = str(msg.get("sensor") or "").upper()  # "A" or "B"
                    now = time.time()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    # optional debounce to suppress burst duplicates
                    if debounce_s and (now - last_fire_ts.get(sensor, 0.0) < debounce_s):
                        continue
                    last_fire_ts[sensor] = now

                    d1 = msg.get("d1") or msg.get("d1_mm") or 0
                    d2 = msg.get("d2") or msg.get("d2_mm") or 0
                    trig_count += 1

                    if sensor == "A":
                        log(f"[Serial] LIDAR A trigger d1={d1} d2={d2} time={ts}")
                        try:
                            barcode_event_queue.put(("lidar_trigger", now))
                        except Exception as e:
                            log(f"[Queue A Error] {e}")

                    elif sensor == "B":
                        log(f"[Serial] LIDAR B trigger d1={d1} d2={d2} time={ts}")
                        try:
                            barcode_event_queue.put(("lidar_trigger", now))
                        except Exception as e:
                            log(f"[Queue B Error] {e}")
                    else:
                        # Unknown sensor label — still useful to see
                        log(f"[Serial] LIDAR ? trigger sensor={sensor} d1={d1} d2={d2}")


            # If _iter_json_lines breaks due to SerialException, we fall through to reconnect.

        except serial.SerialException as e:
            log(f"[Serial Error] {e} — reconnecting in {REOPEN_DELAY:.1f}s")
            time.sleep(REOPEN_DELAY)
        except Exception as e:
            log(f"[Serial Unexpected] {e} — reconnecting in {REOPEN_DELAY:.1f}s")
            time.sleep(REOPEN_DELAY)
        finally:
            try:
                if ser:
                    ser.close()
            except Exception:
                pass

def start_serial_listener(barcode_event_queue: mp.Queue,
                          top_event_queue: mp.Queue):
    """Start the serial listener thread (daemon)."""
    t = threading.Thread(
        target=serial_listener_thread,
        args=(barcode_event_queue, top_event_queue),
        daemon=True
    )
    t.start()
    return t
