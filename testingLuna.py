# pip install pyserial
import serial, time, json, sys

PORT = "COM7"     # <-- set to your Mega's COM port
BAUD = 115200     # <-- matches Serial.begin(...) in your sketch

def open_serial(port, baud):
    # Don’t let flow control mess with things; short timeout for snappy reads
    ser = serial.Serial(port, baud, timeout=0.2, rtscts=False, dsrdtr=False)
    # Mega auto-resets on open; give it time to boot and start printing
    time.sleep(2.0)
    ser.reset_input_buffer()
    return ser

def iter_json_lines(ser):
    """
    Yields parsed JSON dicts from a serial stream.
    Ignores partial/garbage lines; only returns when a full {...} is seen.
    """
    buf = b""
    while True:
        chunk = ser.read(256)
        if not chunk:
            yield None
            continue
        buf += chunk
        while b'\n' in buf:
            line, buf = buf.split(b'\n', 1)
            s = line.decode('utf-8', 'ignore').strip()
            i, j = s.find('{'), s.rfind('}')
            if i != -1 and j != -1 and j > i:
                try:
                    yield json.loads(s[i:j+1])
                except json.JSONDecodeError:
                    # skip malformed lines quietly
                    pass

def mm_to_cm(mm):
    return (mm or 0) / 10.0

def main():
    try:
        ser = open_serial(PORT, BAUD)
        print(f"[Serial] Listening on {PORT} @ {BAUD} … (Ctrl+C to stop)")
    except serial.SerialException as e:
        print(f"[Serial Error] Could not open {PORT}: {e}")
        sys.exit(1)

    hb_count = trig_count = 0
    t0 = time.time()

    try:
        for msg in iter_json_lines(ser):
            if msg is None:
                # no data this tick; you could do other work here
                continue

            evt = (msg.get("event") or "").lower()

            if evt == "hb":
                hb_count += 1
                d1 = msg.get("d1") or msg.get("d1_mm") or 0
                d2 = msg.get("d2") or msg.get("d2_mm") or 0
                near1 = msg.get("near1")
                near2 = msg.get("near2")
                print(f"HB  d1={d1:4d} mm ({mm_to_cm(d1):4.1f} cm)  "
                      f"d2={d2:4d} mm ({mm_to_cm(d2):4.1f} cm)  "
                      f"near1={near1} near2={near2}")
            elif evt == "lidar_trigger":
                trig_count += 1
                sensor = msg.get("sensor")
                d1 = msg.get("d1") or 0
                d2 = msg.get("d2") or 0
                print(f"TRIGGER {sensor}  d1={d1} mm ({mm_to_cm(d1):.1f} cm)  "
                      f"d2={d2} mm ({mm_to_cm(d2):.1f} cm)")
                
            else:
                # Unknown events are fine; ignore silently or print for debugging
                # print(msg)
                pass

            # Optional: simple rate display every ~5s
            now = time.time()
            if now - t0 > 5:
                print(f"[stats] heartbeats={hb_count}  triggers={trig_count}  over {now - t0:.1f}s")
                t0 = now
                hb_count = trig_count = 0

    except KeyboardInterrupt:
        print("\n[Serial] Stopped by user.")
    finally:
        try:
            ser.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
