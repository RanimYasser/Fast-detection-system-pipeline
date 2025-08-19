import time, threading
import numpy as np
from vmbpy import VmbSystem, FrameStatus, VmbFeatureError

class SoftTriggerGrabber:
    """
    Keep the camera streaming; call .fire() to issue a software trigger and get 1 frame.
    Uses callback signature (camera, stream, frame).
    """
    def __init__(self, camera_id, pixel_format='BGR8', buffers=6):
        self.want_one = False
        self.got_one_evt = threading.Event()
        self.last_frame = None

        self.vimba = VmbSystem.get_instance()
        self.vimba.__enter__()

        self.cam = self.vimba.get_camera_by_id(camera_id)
        self.cam.__enter__()
        f = self.cam.get_feature_by_name

        # --- Force the device to IDLE in case a previous run crashed/terminated ---
        self._force_idle(f)

        # --- Basic image/stream config (add more as needed) ---
        for name, val in [('ExposureAuto','Off'), ('GainAuto','Off')]:
            try: f(name).set(val)
            except: pass
        try: f('ExposureTime').set(2000.0)  # us
        except: pass
        try: f('Gain').set(4.0)
        except: pass
        try: f('PixelFormat').set(pixel_format)  # 'BGR8' or 'Mono8'
        except: pass
        try: f('StreamBufferHandlingMode').set('NewestOnly')
        except: pass
        try: f('AcquisitionFrameRateEnable').set(False)  # avoid hidden pacing
        except: pass
        try: f('ExposureMode').set('Timed')
        except: pass
        try: f('AcquisitionMode').set('Continuous')
        except: pass

        # --- Trigger config: OFF -> set Source/Activation -> ON (safe order) ---
        f('TriggerSelector').set('FrameStart')
        try: f('TriggerMode').set('Off')
        except: pass
        f('TriggerSource').set('Software')
        try:
            if f('TriggerActivation').is_writeable():
                f('TriggerActivation').set('RisingEdge')
        except: pass
        f('TriggerMode').set('On')   # enable only after configuring source

        # --- Start streaming after configuration is complete ---
        self.cam.start_streaming(self._on_frame, buffer_count=buffers)
        try:
            f('AcquisitionStart').run()
        except: pass

    def _force_idle(self, f):
        """Best-effort: stop acquisition/streaming and unlock TL before (re)config."""
        # Stop previous acquisition if any
        for cmd in ('AcquisitionStop', 'AcquisitionAbort'):
            try: f(cmd).run()
            except: pass
        # Stop streaming if previous run left it open
        try: self.cam.stop_streaming()
        except: pass
        # Unlock TL params if present
        try:
            tl = f('TLParamsLocked')
            if tl.is_writeable():
                tl.set(0)
        except: pass
        # Give the device a breath; some models need a few ms to clear locks
        time.sleep(0.02)

        # As a last resort, you may uncomment a device reset (slower):
        # try:
        #     f('DeviceReset').run()
        #     time.sleep(2.0)    # wait for reboot
        # except: pass

    def _on_frame(self, cam, stream, frame):
        # Always requeue first to keep the pipeline full/low-latency
        try: stream.queue_frame(frame)
        except: pass
        if frame.get_status() != FrameStatus.Complete:
            return
        if self.want_one:
            img = frame.as_numpy_ndarray()
            self.last_frame = np.ascontiguousarray(img)  # detach
            self.want_one = False
            self.got_one_evt.set()

    def fire(self, timeout=0.3):
        """Issue software trigger and block until the next frame (or timeout)."""
        self.got_one_evt.clear()
        self.want_one = True
        self.cam.get_feature_by_name('TriggerSoftware').run()
        if not self.got_one_evt.wait(timeout):
            self.want_one = False
            return None
        return self.last_frame

    def close(self):
        f = self.cam.get_feature_by_name
        try: f('AcquisitionStop').run()
        except: pass
        try: self.cam.stop_streaming()
        except: pass
        self.cam.__exit__(None, None, None)
        self.vimba.__exit__(None, None, None)

    def discover_av_cameras():
        """Return list of Vimba camera IDs (strings)."""
        with VmbSystem.get_instance() as v:
            cams = v.get_all_cameras()
            return [c.get_id() for c in cams]
