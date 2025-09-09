# luna_serial_trigger_monitor.py
import time, argparse, statistics, serial

def args():
    p = argparse.ArgumentParser("TF-Luna crossing + predicted trigger times (no camera)")
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--rate", type=int, default=200, help="Set Luna output rate (Hz), 0=skip")
    p.add_argument("--base_samples", type=int, default=200)
    p.add_argument("--thresh_cm", type=int, default=6)
    p.add_argument("--hyst_cm", type=int, default=3)
    p.add_argument("--events", type=int, default=100)
    p.add_argument("--L", type=float, required=True, help="Sensor→camera distance (mm)")
    p.add_argument("--V", type=float, required=True, help="Belt speed (mm/s)")
    return p.parse_args()

def set_rate(ser, hz):
    LL, HH = hz & 0xFF, (hz >> 8) & 0xFF
    ser.write(bytes([0x5A,0x06,0x03,LL,HH,(0x5A+0x06+0x03+LL+HH)&0xFF])); ser.flush()

def frames(ser):
    while True:
        b=ser.read(1)
        if not b or b[0]!=0x59: continue
        b2=ser.read(1)
        if not b2 or b2[0]!=0x59: continue
        payload=ser.read(7)
        if len(payload)!=7: continue
        chk=ser.read(1)
        if not chk: continue
        if ((0x59+0x59+sum(payload))&0xFF)!=chk[0]: continue
        dist_cm = payload[0] | (payload[1]<<8)
        yield dist_cm, time.perf_counter_ns()

def main():
    A=args()
    ser=serial.Serial(A.port, A.baud, timeout=0.2)
    time.sleep(0.1)
    if A.rate>0: set_rate(ser, A.rate)

    # learn baseline
    acc=n=0
    for d,t in frames(ser):
        acc+=d; n+=1
        if n>=A.base_samples: break
    base = acc//max(1,n) if n else 100
    target  = max(0, base - A.thresh_cm)
    thr_on  = max(0, base - A.thresh_cm - A.hyst_cm)
    thr_off = max(0, base - A.thresh_cm + A.hyst_cm)

    DELAY_US = 1e6*(A.L/A.V)
    print(f"Base(cm)={base}  Rate(Hz)={A.rate}  DELAY_US={int(DELAY_US)}")
    print("idx, d_prev, d_curr,  t_cross(ms),   t_fire(ms),  naive_jitter(us)")

    prev_d=prev_t=None
    in_box=False; events=0; jit=[]
    t0=time.perf_counter_ns()

    for d,t in frames(ser):
        if prev_d is None:
            prev_d,prev_t=d,t; continue

        now_in = (d <= thr_on)
        if (not in_box) and now_in:
            den = (prev_d - d)
            if den==0:
                t_cross=t
            else:
                a=(prev_d - target)/den
                a=0.0 if a<0 else (1.0 if a>1.0 else a)
                t_cross = int(prev_t + a*(t - prev_t))
            jitter_us = (t - t_cross)/1000.0
            jit.append(jitter_us)
            t_fire = t_cross + int(DELAY_US*1000.0)  # us->ns
            print(f"{events:03d}, {prev_d:5d}, {d:5d}, {(t_cross-t0)/1e6:10.3f}, {(t_fire-t0)/1e6:10.3f}, {jitter_us:10.1f}")
            events+=1; in_box=True
            if events>=A.events: break
        if in_box and d>thr_off: in_box=False
        prev_d,prev_t=d,t

    ser.close()
    if jit:
        mn=min(jit); mx=max(jit); mean=sum(jit)/len(jit); sd=statistics.pstdev(jit) if len(jit)>1 else 0.0
        print("\n=== Summary ===")
        print(f"Events={len(jit)}  naive_jitter_min={mn:.1f} us  max={mx:.1f} us  mean={mean:.1f} us  sd={sd:.1f} us")
        print("Max near one frame (~5,000 us @200 Hz) = sampling quantization; interpolation removes it.")
    else:
        print("No edges captured. Adjust --thresh_cm/--hyst_cm and ensure boxes pass under the beam.")

if __name__=="__main__":
    main()
