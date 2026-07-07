"""
plant_detector.py
------------------
Lightweight, training-free localizer that finds the most "plant-like"
region of a frame so we can draw a bounding box around it and crop it
for embedding extraction.

Approach: HSV color thresholding (greens + a wide hue/saturation band
to catch flowers, brown stems, pots, etc. is intentionally NOT used --
we stay strict to green foliage since that's the most reliable plant
signal) combined with contour detection to find the largest plausible
plant-shaped blob.

This is a heuristic, not a trained object detector. It works well for
"a single plant roughly centered in frame, e.g. on a desk or windowsill",
which matches the registration/recognition workflow described in the
project spec. For tighter, multi-object, species-agnostic detection,
swap this module for a fine-tuned YOLOv8 model (see README).
"""

import cv2
import numpy as np

import config


class PlantDetector:
    def __init__(self):
        # HSV range tuned for typical green foliage under normal indoor/
        # outdoor lighting. Adjust if your plants are unusually pale,
        # variegated, or you're shooting under strong color casts.
        self.lower_green = np.array([25, 35, 35])
        self.upper_green = np.array([95, 255, 255])

    def is_plant_like(self, frame_bgr: np.ndarray) -> bool:
        """Return True only when the frame contains a meaningful amount of green plant-like signal."""
        if frame_bgr is None or frame_bgr.size == 0:
            return False

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        green_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return green_ratio >= config.MIN_GREEN_PIXEL_RATIO

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns (x, y, w, h) bounding box of the most plant-like region,
        or None if nothing sufficiently plant-like was found.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h_frame, w_frame = frame_bgr.shape[:2]
        frame_area = h_frame * w_frame

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        green_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if green_ratio < config.MIN_GREEN_PIXEL_RATIO:
            return None

        # Clean up noise: erode away tiny specks, dilate to reconnect
        # leaf clusters into one solid blob.
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < config.MIN_CONTOUR_AREA_RATIO * frame_area:
            # Nothing big enough to confidently call "a plant" -- caller
            # can fall back to using the full frame instead.
            return None

        x, y, w, h = cv2.boundingRect(largest)

        # Pad the box slightly so we don't crop right at the leaf edge,
        # which keeps a bit of context for the embedding model.
        pad_w, pad_h = int(w * 0.08), int(h * 0.08)
        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(w_frame - x, w + 2 * pad_w)
        h = min(h_frame - y, h + 2 * pad_h)

        return (x, y, w, h)

    @staticmethod
    def crop(frame_bgr: np.ndarray, box):
        """Safely crop a frame to a (x, y, w, h) box, or return the full frame if box is None."""
        if box is None:
            return frame_bgr
        x, y, w, h = box
        return frame_bgr[y : y + h, x : x + w]
