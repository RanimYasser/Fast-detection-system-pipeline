# sensor_monitor.py
# Read switch events from Arduino over serial and print/log them.
# Works with your sketch that prints lines like:
#   "D2: PRESSED", "D2: RELEASED", "D3: PRESSED", "D3: RELEASED"

import sys, time, argparse
from datetime import datetime

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("Install pyserial first:  pip install pyserial")
    sys.exit(1)

def pick_port(preferred: str | None) -> str | None:
    if preferred:
        return preferred
    candidates = []

    for p in list_ports.comports():
        manu = (p.manufacturer or "").lower()
        desc = (p.description or "").lower()
        # Common Arduino/USB-serial VID/PID or names (Arduino, CH340, FTDI, RP2040)
        if (
            "arduino" in manu or "arduino" in desc or
            "wch" in manu or "ch340" in desc or
            "ftdi" in manu or "usb serial" in desc or
            p.vid in {0x2341, 0x1A86, 0x0403, 0x2E8A}
        ):
            candidates.append(p.device)

    if candidates:
        return candidates[0]

    # Fallback: first serial-ish device
    all_ports = [p.device for p in list_ports.comports()]
    for dev in all_ports:
        if sys.platform.startswith("win") and dev.upper().startswith("COM"):
            return dev
        if not sys.platform.startswith("win") and ("/dev/ttyACM" in dev or "/dev/ttyUSB" in dev):
            return dev

    return None

def main():
    ap = argparse.ArgumentParser(description="Monitor D2/D3 switch events from Arduino.")
    ap.add_argument("--port", help="Serial port (e.g., COM5, /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("--csv", help="Optional path to log CSV (timestamp,event)")
    args = ap.parse_args()

    port = pick_port(args.port)
    if not port:
        print("No serial port found. Pass one explicitly with --port (e.g., --port COM5).")
        sys.exit(2)

    print(f"Opening {port} @ {args.baud} ... (close Arduino Serial Monitor first)")
    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as e:
        print("Could not open serial:", e)
        sys.exit(3)

    # Give Arduino time to reset after opening port
    time.sleep(2.0)
    ser.reset_input_buffer()

    writer = None
    if args.csv:
        import csv
        f = open(args.csv, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "event"])

    print("Listening. Press Ctrl+C to quit.\n")
    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if not text:
                continue

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Expected: "D2: PRESSED", "D3: RELEASED", etc.
            print(f"[{ts}] {text}")
            if writer:
                writer.writerow([ts, text])
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        if writer:
            f.close()

if __name__ == "__main__":
    main()
