"""
camera_manager.py
------------------
Unified camera interface that works identically whether the frame
source is:
  - a local PC webcam (integer device index, e.g. 0)
  - a phone running IP Webcam / DroidCam (MJPEG stream over HTTP)
  - an ESP32-CAM (MJPEG stream over HTTP)

All three are actually the same thing from OpenCV's point of view --
cv2.VideoCapture() accepts either an integer index or a URL string and
handles MJPEG/HTTP streams transparently. This class just wraps that
in a small, threaded, fail-safe API so the GUI never blocks waiting on
a slow/dropped frame.
"""

import threading
import time
import cv2

import config


class CameraManager:
    """
    Usage:
        cam = CameraManager()
        cam.open("webcam", index=0)
            # or
        cam.open("ip_webcam", url="http://192.168.1.50:8080/video")
            # or
        cam.open("esp32cam", url="http://192.168.1.60/stream")

        cam.start()
        frame = cam.get_frame()   # latest frame, or None
        cam.stop()
    """

    SOURCE_TYPES = ("webcam", "ip_webcam", "droidcam", "esp32cam")

    def __init__(self):
        self.cap = None
        self.source_type = None
        self.source_value = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._last_error = None

    def open(self, source_type: str, index: int = None, url: str = None):
        """Configure (but don't yet start reading from) a camera source."""
        if source_type not in self.SOURCE_TYPES:
            raise ValueError(
                f"Unknown source_type '{source_type}'. Must be one of {self.SOURCE_TYPES}"
            )

        self.source_type = source_type
        self.source_value = index if source_type == "webcam" else url

        if source_type == "webcam":
            device_index = index if index is not None else config.DEFAULT_WEBCAM_INDEX
            try:
                self.cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            except Exception:
                self.cap = cv2.VideoCapture(device_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # IP Webcam / DroidCam / ESP32-CAM are all plain MJPEG-over-HTTP
            # streams as far as OpenCV is concerned -- just point VideoCapture
            # at the URL.
            if not url:
                raise ValueError(
                    f"A stream URL is required for source_type='{source_type}'"
                )
            self.cap = cv2.VideoCapture(url)

        if not self.cap.isOpened():
            self._last_error = (
                f"Could not open camera source: {source_type} ({self.source_value})"
            )
            return False

        self._last_error = None
        return True

    def start(self):
        """Begin the background frame-grabbing thread."""
        if self.cap is None or not self.cap.isOpened():
            return False
        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()
        return True

    def _update_loop(self):
        consecutive_failures = 0
        while self._running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    self._last_error = "Lost connection to camera stream."
                    self._running = False
                    break
                time.sleep(0.05)

    def get_frame(self):
        """Returns the most recent frame (BGR numpy array), or None if unavailable."""
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def is_running(self) -> bool:
        return self._running

    def get_last_error(self):
        return self._last_error

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._latest_frame = None
