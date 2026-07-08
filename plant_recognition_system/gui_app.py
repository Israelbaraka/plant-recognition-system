"""
gui_app.py
----------
Modern Tkinter GUI tying together the camera manager, plant detector,
feature extractor, recognition engine, and database into one
desktop application with two modes: Registration and Recognition.

Layout:
    +---------------------------------------------------+
    |  [ Live Camera Preview (with bounding box) ]       |
    |                                                     |
    +---------------------------------------------------+
    |  Captured image thumbnails (registration)          |
    +-----------------------+----------------------------+
    | Camera & mode controls | Registered plants list     |
    | Plant name / capture   | Delete button               |
    | Register / Recognize   | Status / confidence panel   |
    +-----------------------+----------------------------+
"""

import os
import re
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

import config
from database import PlantDatabase
from feature_extractor import FeatureExtractor
from botanical_classifier import BotanicalClassifier
from plant_detector import PlantDetector
from recognition_engine import RecognitionEngine
from camera_manager import CameraManager

# ----------------------------------------------------------------------
# Color palette (modern dark-ish theme)
# ----------------------------------------------------------------------
COLOR_BG = "#1e1f29"
COLOR_PANEL = "#272935"
COLOR_ACCENT = "#4caf7d"
COLOR_ACCENT_DARK = "#3a8a61"
COLOR_DANGER = "#e05c5c"
COLOR_TEXT = "#f1f1f4"
COLOR_SUBTEXT = "#9c9eb0"
COLOR_CANVAS_BG = "#0d0e14"

FONT_HEADER = ("Segoe UI", 14, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_BODY_BOLD = ("Segoe UI", 10, "bold")
FONT_STATUS = ("Segoe UI", 16, "bold")


class PlantRecognitionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Plant Recognition System")
        self.root.geometry("1180x780")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(1000, 700)

        # ---------------- Backend components ----------------
        self.db = PlantDatabase()
        self.extractor = FeatureExtractor()
        self.botanical_classifier = BotanicalClassifier()
        self.detector = PlantDetector()
        self.engine = RecognitionEngine(self.db)
        self.camera = CameraManager()

        # ---------------- Runtime state ----------------
        self.mode = "idle"  # "idle" | "registration" | "recognition"
        self.frame_counter = 0
        self.captured_images = []  # list of np.ndarray (BGR) captured this session
        self.capture_target = 0
        self.capturing = False
        self.last_box = None
        self.last_label = None
        self.last_color = COLOR_SUBTEXT
        self.thumbnail_refs = []  # keep PhotoImage refs alive
        self._camera_start_worker = None
        self._recognition_worker = None
        self._recognition_lock = threading.Lock()
        self._last_recognition_launch = 0.0
        self._auto_register_lock = threading.Lock()
        self._pending_auto_registrations = set()

        self._build_style()
        self._build_layout()
        self._refresh_plant_list()
        self._update_preview_loop()

    # ====================================================================
    # STYLE
    # ====================================================================
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=COLOR_PANEL)
        style.configure("Root.TFrame", background=COLOR_BG)
        style.configure(
            "TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=FONT_BODY
        )
        style.configure(
            "Header.TLabel",
            background=COLOR_PANEL,
            foreground=COLOR_TEXT,
            font=FONT_HEADER,
        )
        style.configure(
            "Sub.TLabel",
            background=COLOR_PANEL,
            foreground=COLOR_SUBTEXT,
            font=FONT_BODY,
        )
        style.configure(
            "Status.TLabel",
            background=COLOR_PANEL,
            foreground=COLOR_TEXT,
            font=FONT_STATUS,
        )

        style.configure(
            "TButton",
            font=FONT_BODY_BOLD,
            padding=8,
            background=COLOR_ACCENT,
            foreground="#10130f",
            borderwidth=0,
        )
        style.map("TButton", background=[("active", COLOR_ACCENT_DARK)])

        style.configure(
            "Danger.TButton",
            font=FONT_BODY_BOLD,
            padding=8,
            background=COLOR_DANGER,
            foreground="#1a0d0d",
            borderwidth=0,
        )
        style.map("Danger.TButton", background=[("active", "#b94747")])

        style.configure("TCombobox", padding=4)
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=COLOR_CANVAS_BG,
            background=COLOR_ACCENT,
        )

    # ====================================================================
    # LAYOUT
    # ====================================================================
    def _build_layout(self):
        outer_frame = ttk.Frame(self.root, style="Root.TFrame")
        outer_frame.pack(fill="both", expand=True, padx=12, pady=12)

        content_canvas = tk.Canvas(
            outer_frame,
            bg=COLOR_BG,
            highlightthickness=0,
            borderwidth=0,
        )
        content_scrollbar = ttk.Scrollbar(
            outer_frame, orient="vertical", command=content_canvas.yview
        )
        content_canvas.configure(yscrollcommand=content_scrollbar.set)
        content_scrollbar.pack(side="right", fill="y")
        content_canvas.pack(side="left", fill="both", expand=True)

        root_frame = ttk.Frame(content_canvas, style="Root.TFrame")
        root_window = content_canvas.create_window(
            (0, 0), window=root_frame, anchor="nw"
        )

        def _sync_scrollregion(event=None):
            content_canvas.configure(scrollregion=content_canvas.bbox("all"))

        def _sync_frame_width(event):
            content_canvas.itemconfigure(root_window, width=event.width)

        root_frame.bind("<Configure>", _sync_scrollregion)
        content_canvas.bind("<Configure>", _sync_frame_width)

        # ---- Top: always-visible camera toolbar ----
        # Some window sizes can clip the bottom control panel, so keep the
        # camera start/stop actions up here where they are impossible to miss.
        self.source_var = tk.StringVar(value="webcam")
        self.source_value_var = tk.StringVar(value=str(config.DEFAULT_WEBCAM_INDEX))

        camera_bar = ttk.Frame(root_frame, style="TFrame")
        camera_bar.pack(fill="x", pady=(0, 10))

        ttk.Label(camera_bar, text="Camera", style="Header.TLabel").pack(
            side="left", padx=(12, 10)
        )
        source_combo = ttk.Combobox(
            camera_bar,
            textvariable=self.source_var,
            state="readonly",
            width=14,
            values=["webcam", "ip_webcam", "droidcam", "esp32cam"],
        )
        source_combo.pack(side="left", padx=(0, 8), pady=10)
        source_combo.bind("<<ComboboxSelected>>", self._on_source_type_changed)

        ttk.Entry(camera_bar, textvariable=self.source_value_var, width=28).pack(
            side="left", padx=(0, 8)
        )
        self.start_camera_btn = ttk.Button(
            camera_bar, text="Start Camera", command=self._start_camera
        )
        self.start_camera_btn.pack(side="left", padx=(0, 6))
        self.stop_camera_btn = ttk.Button(
            camera_bar, text="Stop Camera", command=self._stop_camera
        )
        self.stop_camera_btn.pack(side="left")

        action_bar = ttk.Frame(root_frame, style="TFrame")
        action_bar.pack(fill="x", pady=(0, 10))

        ttk.Label(action_bar, text="Plants", style="Header.TLabel").pack(
            side="left", padx=(12, 10)
        )
        self.delete_selected_top_btn = ttk.Button(
            action_bar,
            text="Delete Selected",
            style="Danger.TButton",
            command=self._delete_selected_plant,
        )
        self.delete_selected_top_btn.pack(side="left", padx=(0, 6))
        self.delete_all_top_btn = ttk.Button(
            action_bar,
            text="Delete All",
            style="Danger.TButton",
            command=self._delete_all_plants,
        )
        self.delete_all_top_btn.pack(side="left")

        # ---- Top: video preview ----
        video_frame = ttk.Frame(root_frame, style="Root.TFrame")
        video_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            video_frame,
            bg=COLOR_CANVAS_BG,
            width=config.CAMERA_FRAME_WIDTH,
            height=config.CAMERA_FRAME_HEIGHT,
            highlightthickness=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 12))

        # ---- Right: status + plant list ----
        right_panel = ttk.Frame(video_frame, width=300, style="TFrame")
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="Recognition Status", style="Header.TLabel").pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        self.status_label = ttk.Label(right_panel, text="Idle", style="Status.TLabel")
        self.status_label.pack(anchor="w", padx=12)

        botanical_source = self._botanical_source_label()
        self.botanical_mode_label = ttk.Label(
            right_panel,
            text=botanical_source,
            style="Sub.TLabel",
            wraplength=260,
            justify="left",
        )
        self.botanical_mode_label.pack(anchor="w", padx=12, pady=(4, 0))

        self.confidence_var = tk.DoubleVar(value=0.0)
        ttk.Label(right_panel, text="Confidence", style="Sub.TLabel").pack(
            anchor="w", padx=12, pady=(10, 2)
        )
        self.confidence_bar = ttk.Progressbar(
            right_panel,
            orient="horizontal",
            maximum=100,
            variable=self.confidence_var,
            length=260,
        )
        self.confidence_bar.pack(anchor="w", padx=12)
        self.confidence_label = ttk.Label(right_panel, text="0.0%", style="Sub.TLabel")
        self.confidence_label.pack(anchor="w", padx=12, pady=(2, 12))

        ttk.Separator(right_panel).pack(fill="x", padx=12, pady=6)

        ttk.Label(right_panel, text="Registered Plants", style="Header.TLabel").pack(
            anchor="w", padx=12, pady=(6, 4)
        )

        list_container = ttk.Frame(right_panel)
        list_container.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        self.plant_listbox = tk.Listbox(
            list_container,
            bg=COLOR_CANVAS_BG,
            fg=COLOR_TEXT,
            selectbackground=COLOR_ACCENT,
            selectforeground="#0d0e14",
            font=FONT_BODY,
            borderwidth=0,
            highlightthickness=0,
        )
        self.plant_listbox.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(
            list_container, orient="vertical", command=self.plant_listbox.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.plant_listbox.configure(yscrollcommand=scrollbar.set)
        self.plant_listbox.bind("<<ListboxSelect>>", self._on_plant_selected)

        self.plant_detail_label = ttk.Label(
            right_panel, text="", style="Sub.TLabel", wraplength=260, justify="left"
        )
        self.plant_detail_label.pack(anchor="w", padx=12, pady=(0, 6))

        self.delete_selected_btn = ttk.Button(
            right_panel,
            text="Delete Selected Plant",
            style="Danger.TButton",
            command=self._delete_selected_plant,
        )
        self.delete_selected_btn.pack(fill="x", padx=12, pady=(0, 8))

        self.delete_all_btn = ttk.Button(
            right_panel,
            text="Delete All Plants",
            style="Danger.TButton",
            command=self._delete_all_plants,
        )
        self.delete_all_btn.pack(fill="x", padx=12, pady=(0, 12))

        # ---- Middle: image thumbnails ----
        self.thumb_frame = ttk.Frame(root_frame, style="TFrame")
        self.thumb_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(
            self.thumb_frame, text="Captured Images Preview", style="Sub.TLabel"
        ).pack(anchor="w", padx=12, pady=(8, 2))
        self.thumb_strip = ttk.Frame(self.thumb_frame, style="TFrame")
        self.thumb_strip.pack(fill="x", padx=12, pady=(0, 8))

        # ---- Bottom: controls ----
        controls = ttk.Frame(root_frame, style="TFrame")
        controls.pack(fill="x", pady=(10, 0))

        # Camera source controls
        cam_box = ttk.Frame(controls, style="TFrame")
        cam_box.pack(side="left", fill="y", padx=(12, 24), pady=10)

        ttk.Label(cam_box, text="Camera Source", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        source_combo = ttk.Combobox(
            cam_box,
            textvariable=self.source_var,
            state="readonly",
            width=18,
            values=["webcam", "ip_webcam", "droidcam", "esp32cam"],
        )
        source_combo.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        source_combo.bind("<<ComboboxSelected>>", self._on_source_type_changed)

        ttk.Label(cam_box, text="Index / URL:", style="Sub.TLabel").grid(
            row=2, column=0, sticky="w", pady=(8, 2)
        )
        self.source_value_entry = ttk.Entry(
            cam_box, textvariable=self.source_value_var, width=28
        )
        self.source_value_entry.grid(row=3, column=0, columnspan=2, sticky="w")

        btn_row = ttk.Frame(cam_box, style="TFrame")
        btn_row.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Button(btn_row, text="Start Camera", command=self._start_camera).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(btn_row, text="Stop Camera", command=self._stop_camera).pack(
            side="left"
        )

        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", pady=10)

        # Registration controls
        reg_box = ttk.Frame(controls, style="TFrame")
        reg_box.pack(side="left", fill="y", padx=24, pady=10)

        ttk.Label(reg_box, text="Plant Registration", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        ttk.Label(reg_box, text="Plant Name:", style="Sub.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        self.plant_name_var = tk.StringVar()
        ttk.Entry(reg_box, textvariable=self.plant_name_var, width=24).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        reg_btn_row = ttk.Frame(reg_box, style="TFrame")
        reg_btn_row.grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(
            reg_btn_row, text="Capture Images", command=self._start_capture_burst
        ).pack(side="left", padx=(0, 6))
        ttk.Button(reg_btn_row, text="Save Images", command=self._register_plant).pack(
            side="left", padx=(0, 6)
        )

        self.capture_progress_label = ttk.Label(
            reg_box, text="0 images captured", style="Sub.TLabel"
        )
        self.capture_progress_label.grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", pady=10)

        # Recognition controls
        rec_box = ttk.Frame(controls, style="TFrame")
        rec_box.pack(side="left", fill="y", padx=24, pady=10)

        ttk.Label(rec_box, text="Plant Recognition", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        self.recognize_btn = ttk.Button(
            rec_box, text="Start Recognition", command=self._toggle_recognition
        )
        self.recognize_btn.grid(row=1, column=0, sticky="w", pady=(0, 8))

        ttk.Label(rec_box, text="Confidence Threshold", style="Sub.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        self.threshold_var = tk.DoubleVar(value=config.BOTANICAL_CONFIDENCE_THRESHOLD)
        threshold_slider = ttk.Scale(
            rec_box,
            from_=0.4,
            to=0.95,
            orient="horizontal",
            variable=self.threshold_var,
            length=200,
            command=self._on_threshold_changed,
        )
        threshold_slider.grid(row=3, column=0, sticky="w")
        self.threshold_value_label = ttk.Label(
            rec_box,
            text=f"{config.BOTANICAL_CONFIDENCE_THRESHOLD:.2f}",
            style="Sub.TLabel",
        )
        self.threshold_value_label.grid(row=4, column=0, sticky="w")

    # ====================================================================
    # CAMERA CONTROL
    # ====================================================================
    def _on_source_type_changed(self, event=None):
        defaults = {
            "webcam": str(config.DEFAULT_WEBCAM_INDEX),
            "ip_webcam": config.DEFAULT_IP_WEBCAM_URL,
            "droidcam": config.DEFAULT_DROIDCAM_URL,
            "esp32cam": config.DEFAULT_ESP32CAM_URL,
        }
        self.source_value_var.set(defaults.get(self.source_var.get(), ""))

    def _start_camera(self):
        if (
            self._camera_start_worker is not None
            and self._camera_start_worker.is_alive()
        ):
            return

        self.start_camera_btn.configure(state="disabled")
        self.stop_camera_btn.configure(state="normal")
        self._set_status("Starting camera...", COLOR_SUBTEXT)

        source_type = self.source_var.get()
        value = self.source_value_var.get().strip()

        def worker():
            try:
                self.camera.stop()

                if source_type == "webcam":
                    try:
                        index = int(value)
                    except ValueError:
                        raise ValueError("Webcam index must be an integer (e.g. 0).")
                    ok = self.camera.open("webcam", index=index)
                else:
                    if not value:
                        raise ValueError("Please provide a stream URL.")
                    ok = self.camera.open(source_type, url=value)

                if not ok:
                    raise RuntimeError(
                        self.camera.get_last_error() or "Could not open camera."
                    )

                self.camera.start()
                self.root.after(0, lambda: self._on_camera_started())
            except Exception as exc:
                self.root.after(0, lambda: self._on_camera_start_failed(str(exc)))

        self._camera_start_worker = threading.Thread(target=worker, daemon=True)
        self._camera_start_worker.start()

    def _on_camera_started(self):
        self.start_camera_btn.configure(state="normal")
        self.stop_camera_btn.configure(state="normal")
        self._set_status("Camera started", COLOR_ACCENT)

    def _on_camera_start_failed(self, error_text):
        self.start_camera_btn.configure(state="normal")
        self.stop_camera_btn.configure(state="normal")
        messagebox.showerror("Camera Error", error_text)
        self._set_status("Camera stopped", COLOR_DANGER)

    def _stop_camera(self):
        self.camera.stop()
        self.canvas.delete("all")
        self._set_status("Camera stopped", COLOR_SUBTEXT)
        self.start_camera_btn.configure(state="normal")

    # ====================================================================
    # PREVIEW / RECOGNITION LOOP
    # ====================================================================
    def _update_preview_loop(self):
        frame = self.camera.get_frame()
        if frame is not None:
            self.frame_counter += 1
            display_frame = frame.copy()

            if self.mode == "recognition":
                self._draw_box(
                    display_frame, self.last_box, self.last_color, self.last_label
                )
            elif self.mode == "registration":
                box = self.detector.detect(frame)
                self._draw_box(display_frame, box, COLOR_ACCENT, None)

            if self.mode == "recognition":
                self._maybe_launch_recognition_worker(frame)

            self._render_to_canvas(display_frame)

        self.root.after(config.PREVIEW_REFRESH_MS, self._update_preview_loop)

    def _maybe_launch_recognition_worker(self, frame):
        # Keep the UI thread light by doing detection / embedding / match
        # work in the background. Only one worker runs at a time.
        if self._recognition_worker is not None and self._recognition_worker.is_alive():
            return

        now = time.monotonic()
        if (
            now - self._last_recognition_launch
            < (config.RECOGNITION_FRAME_SKIP * config.PREVIEW_REFRESH_MS) / 1000.0
        ):
            return

        self._last_recognition_launch = now
        frame_snapshot = frame.copy()

        self._recognition_worker = threading.Thread(
            target=self._run_recognition_step,
            args=(frame_snapshot,),
            daemon=True,
        )
        self._recognition_worker.start()

    def _run_recognition_step(self, frame):
        # Only run the relatively expensive pipeline in a worker thread.

        scale = max(0.35, min(1.0, float(config.RECOGNITION_ANALYSIS_SCALE)))
        if scale < 0.999:
            analysis_frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            analysis_frame = frame

        if not self.detector.is_plant_like(analysis_frame):

            def clear_ui():
                self.last_box = None
                self.last_label = "No plant detected"
                self.last_color = COLOR_DANGER
                self._set_status("No plant detected", COLOR_DANGER)
                self.confidence_var.set(0.0)
                self.confidence_label.configure(text="0.0%")

            self.root.after(0, clear_ui)
            return

        box = self.detector.detect(analysis_frame)
        if box is None:

            def clear_ui_box():
                self.last_box = None
                self.last_label = "No plant detected"
                self.last_color = COLOR_DANGER
                self._set_status("No plant detected", COLOR_DANGER)
                self.confidence_var.set(0.0)
                self.confidence_label.configure(text="0.0%")

            self.root.after(0, clear_ui_box)
            return

        crop = self.detector.crop(analysis_frame, box)

        display_box = None
        if box is not None and scale < 0.999:
            x, y, w, h = box
            inv = 1.0 / scale
            display_box = (
                int(x * inv),
                int(y * inv),
                int(w * inv),
                int(h * inv),
            )
        else:
            display_box = box

        botanical_result = self.botanical_classifier.classify(crop)
        if botanical_result is not None:
            self.root.after(
                0,
                lambda: self._show_botanical_result(
                    botanical_result, display_box, crop.copy()
                ),
            )
            return

        try:
            embedding = self.extractor.extract(crop)
        except ValueError:
            return

        result = self.engine.recognize(embedding)

        def update_ui():
            self.last_box = display_box
            if result.is_known:
                self.last_label = f"{result.name} ({result.confidence:.1f}%)"
                self.last_color = COLOR_ACCENT
                self._set_status(f"Recognized: {result.name}", COLOR_ACCENT)
            else:
                self.last_label = f"Unknown Plant ({result.confidence:.1f}%)"
                self.last_color = COLOR_DANGER
                self._set_status("Unknown Plant", COLOR_DANGER)

            self.confidence_var.set(result.confidence)
            self.confidence_label.configure(text=f"{result.confidence:.1f}%")

        self.root.after(0, update_ui)

    def _show_botanical_result(self, result, box, crop=None):
        self.last_box = box
        is_confident = result.is_confident
        label_name = result.scientific_name or result.name
        label = f"{label_name} ({result.confidence:.1f}%)"

        if is_confident:
            self.last_label = label
            self.last_color = COLOR_ACCENT
            self._set_status(f"Species: {label_name}", COLOR_ACCENT)
        else:
            self.last_label = f"Uncertain species ({result.confidence:.1f}%)"
            self.last_color = COLOR_DANGER
            self._set_status("Uncertain botanical result", COLOR_DANGER)

        detail = result.source
        if result.family:
            detail = f"{detail} | Family: {result.family}"
        self.botanical_mode_label.configure(text=detail)
        self.confidence_var.set(result.confidence)
        self.confidence_label.configure(text=f"{result.confidence:.1f}%")

        if (
            crop is not None
            and is_confident
            and self.botanical_classifier.mode == "plantnet_api"
            and config.AUTO_REGISTER_PLANTNET_RESULTS
        ):
            self._maybe_auto_register_botanical_result(result, crop)

    def _draw_box(self, frame, box, color_hex, label):
        if box is None:
            return
        x, y, w, h = box
        bgr = self._hex_to_bgr(color_hex)
        cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
        if label:
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x, y - text_h - 12), (x + text_w + 8, y), bgr, -1)
            cv2.putText(
                frame,
                label,
                (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    @staticmethod
    def _hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def _render_to_canvas(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        canvas_w = self.canvas.winfo_width() or config.CAMERA_FRAME_WIDTH
        canvas_h = self.canvas.winfo_height() or config.CAMERA_FRAME_HEIGHT
        image = image.resize((max(canvas_w, 10), max(canvas_h, 10)))

        self._tk_image = ImageTk.PhotoImage(image=image)  # keep reference alive
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

    # ====================================================================
    # REGISTRATION MODE
    # ====================================================================
    def _start_capture_burst(self):
        if self.camera.get_frame() is None:
            messagebox.showwarning(
                "No Camera", "Start a camera before capturing images."
            )
            return
        name = self.plant_name_var.get().strip()
        if not name:
            messagebox.showwarning(
                "Missing Name", "Enter a plant name before capturing images."
            )
            return
        if self.db.plant_exists(name) and len(self.captured_images) == 0:
            messagebox.showwarning(
                "Duplicate Name",
                f"'{name}' is already registered. Choose a unique plant name.",
            )
            return

        self.mode = "registration"
        self.capturing = True
        self.capture_target = len(self.captured_images) + config.IMAGES_PER_REGISTRATION
        self._set_status(f"Capturing images for '{name}'...", COLOR_ACCENT)
        self._capture_step()

    def _capture_step(self):
        if not self.capturing or len(self.captured_images) >= self.capture_target:
            self.capturing = False
            self._set_status(
                f"Captured {len(self.captured_images)} images. Click 'Save Images' to save.",
                COLOR_ACCENT,
            )
            return

        frame = self.camera.get_frame()
        if frame is not None:
            if not self.detector.is_plant_like(frame):
                self._set_status(
                    "No plant detected. Point the camera at a real plant before capturing.",
                    COLOR_DANGER,
                )
                self.root.after(config.CAPTURE_INTERVAL_MS, self._capture_step)
                return

            box = self.detector.detect(frame)
            if box is None:
                self._set_status(
                    "No plant detected. Point the camera at a real plant before capturing.",
                    COLOR_DANGER,
                )
                self.root.after(config.CAPTURE_INTERVAL_MS, self._capture_step)
                return

            crop = self.detector.crop(frame, box)
            self.captured_images.append(crop.copy())
            self._add_thumbnail(crop)
            self.capture_progress_label.configure(
                text=f"{len(self.captured_images)} images captured"
            )
            self._set_status(
                f"Capturing plant images... ({len(self.captured_images)}/{self.capture_target})",
                COLOR_ACCENT,
            )

        self.root.after(config.CAPTURE_INTERVAL_MS, self._capture_step)

    def _add_thumbnail(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((70, 55))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.thumbnail_refs.append(tk_img)

        # Keep the strip from growing unbounded on screen -- show the
        # most recent thumbnails only.
        if len(self.thumbnail_refs) > 10:
            self.thumbnail_refs.pop(0)
            for child in self.thumb_strip.winfo_children():
                child.destroy()
            for img in self.thumbnail_refs:
                tk.Label(self.thumb_strip, image=img, bg=COLOR_PANEL).pack(
                    side="left", padx=2
                )
        else:
            tk.Label(self.thumb_strip, image=tk_img, bg=COLOR_PANEL).pack(
                side="left", padx=2
            )

    def _register_plant(self):
        name = self.plant_name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing Name", "Enter a plant name.")
            return
        if self.db.plant_exists(name):
            messagebox.showwarning("Duplicate Name", f"'{name}' is already registered.")
            return
        if len(self.captured_images) < config.MIN_IMAGES_TO_REGISTER:
            messagebox.showwarning(
                "Not Enough Images",
                f"Capture at least {config.MIN_IMAGES_TO_REGISTER} images first "
                f"(currently have {len(self.captured_images)}).",
            )
            return

        def worker():
            try:
                plant_dir = self._plant_dataset_dir(name)
                os.makedirs(plant_dir, exist_ok=True)

                plant_id = self.db.add_plant(name)

                for i, image in enumerate(self.captured_images, start=1):
                    image_path = os.path.join(plant_dir, f"image{i}.jpg")
                    cv2.imwrite(image_path, image)
                    embedding = self.extractor.extract(image)
                    self.db.add_embedding(plant_id, image_path, embedding)

                self.engine.refresh_gallery()  # new plant usable immediately

                self.root.after(0, lambda: self._on_registration_complete(name))
            except Exception as exc:
                self.root.after(
                    0, lambda: messagebox.showerror("Registration Failed", str(exc))
                )

        threading.Thread(target=worker, daemon=True).start()
        self._set_status(f"Registering '{name}'...", COLOR_ACCENT)

    def _maybe_auto_register_botanical_result(self, result, crop):
        plant_name = (result.scientific_name or result.name or "").strip()
        if not plant_name:
            return

        if self.db.plant_exists(plant_name):
            return

        with self._auto_register_lock:
            if plant_name in self._pending_auto_registrations:
                return
            self._pending_auto_registrations.add(plant_name)

        self._set_status(f"Auto-registering '{plant_name}'...", COLOR_SUBTEXT)

        worker = threading.Thread(
            target=self._auto_register_botanical_worker,
            args=(plant_name, crop.copy()),
            daemon=True,
        )
        worker.start()

    def _auto_register_botanical_worker(self, name, crop):
        try:
            if self.db.plant_exists(name):
                return

            plant_dir = self._plant_dataset_dir(name)
            os.makedirs(plant_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(plant_dir, f"auto_{timestamp}.jpg")
            ok = cv2.imwrite(image_path, crop)
            if not ok:
                raise RuntimeError("Could not save the auto-registered plant image.")

            embedding = self.extractor.extract(crop)
            plant_id = self.db.add_plant(name)
            self.db.add_embedding(plant_id, image_path, embedding)
            self.engine.refresh_gallery()

            self.root.after(0, lambda: self._on_auto_registration_complete(name))
        except Exception as exc:
            error_text = str(exc)
            self.root.after(
                0,
                lambda: self._set_status(
                    f"Auto-registration failed for '{name}': {error_text}", COLOR_DANGER
                ),
            )
        finally:
            with self._auto_register_lock:
                self._pending_auto_registrations.discard(name)

    def _on_auto_registration_complete(self, name):
        self._refresh_plant_list()
        self._set_status(f"Auto-registered '{name}'", COLOR_ACCENT)

    def _plant_dataset_dir(self, name):
        return os.path.join(config.DATASET_DIR, self._safe_path_component(name))

    @staticmethod
    def _safe_path_component(value):
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
        value = value.strip(" .")
        return value or "unknown_plant"

    def _on_registration_complete(self, name):
        messagebox.showinfo(
            "Plant Registered",
            f"'{name}' was registered with {len(self.captured_images)} images.",
        )
        self.captured_images = []
        self.thumbnail_refs = []
        for child in self.thumb_strip.winfo_children():
            child.destroy()
        self.plant_name_var.set("")
        self.capture_progress_label.configure(text="0 images captured")
        self._set_status("Idle", COLOR_SUBTEXT)
        self.mode = "idle"
        self._refresh_plant_list()

    # ====================================================================
    # RECOGNITION MODE
    # ====================================================================
    def _toggle_recognition(self):
        if self.mode == "recognition":
            self.mode = "idle"
            self.recognize_btn.configure(text="Start Recognition")
            self._set_status("Idle", COLOR_SUBTEXT)
            self.confidence_var.set(0.0)
            self.confidence_label.configure(text="0.0%")
        else:
            if self.camera.get_frame() is None:
                messagebox.showwarning(
                    "No Camera", "Start a camera before recognizing plants."
                )
                return
            if (
                not self.botanical_classifier.is_available()
                and self.engine.gallery_vectors.shape[0] == 0
            ):
                messagebox.showwarning(
                    "No Registered Plants",
                    "Configure a botanical model/API key or register at least one plant.",
                )
                return
            self.mode = "recognition"
            self.recognize_btn.configure(text="Stop Recognition")
            self._set_status("Scanning...", COLOR_SUBTEXT)
            self.botanical_mode_label.configure(text=self._botanical_source_label())

    def _on_threshold_changed(self, value):
        v = float(value)
        self.engine.set_threshold(v)
        config.BOTANICAL_CONFIDENCE_THRESHOLD = v
        self.threshold_value_label.configure(text=f"{v:.2f}")

    # ====================================================================
    # PLANT LIST / DELETE
    # ====================================================================
    def _refresh_plant_list(self):
        self.plant_listbox.delete(0, tk.END)
        self._plant_rows = self.db.get_all_plants()
        for plant in self._plant_rows:
            self.plant_listbox.insert(
                tk.END, f"{plant['name']}  ({plant['num_images']} imgs)"
            )

    def _on_plant_selected(self, event=None):
        sel = self.plant_listbox.curselection()
        if not sel:
            return
        plant = self._plant_rows[sel[0]]
        self.plant_detail_label.configure(
            text=f"Registered: {plant['registered_at']}\nImages: {plant['num_images']}"
        )

    def _delete_selected_plant(self):
        sel = self.plant_listbox.curselection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a plant from the list first.")
            return
        plant = self._plant_rows[sel[0]]
        if not messagebox.askyesno(
            "Confirm Delete", f"Delete '{plant['name']}' and all its images/embeddings?"
        ):
            return

        self.db.delete_plant(plant["id"])
        self.engine.refresh_gallery()  # deleted plant stops matching immediately

        # Remove dataset folder for this plant too (best-effort).
        plant_dir = self._plant_dataset_dir(plant["name"])
        if os.path.isdir(plant_dir):
            import shutil

            shutil.rmtree(plant_dir, ignore_errors=True)

        self._refresh_plant_list()
        self.plant_detail_label.configure(text="")
        self._set_status(f"Deleted '{plant['name']}'", COLOR_SUBTEXT)

    def _delete_all_plants(self):
        if not self._plant_rows:
            messagebox.showinfo(
                "No Plants", "There are no registered plants to delete."
            )
            return
        if not messagebox.askyesno(
            "Confirm Delete All",
            "Delete ALL registered plants, embeddings, and their saved images?",
        ):
            return

        self.db.delete_all_plants()
        self.engine.refresh_gallery()

        if os.path.isdir(config.DATASET_DIR):
            import shutil

            shutil.rmtree(config.DATASET_DIR, ignore_errors=True)
        os.makedirs(config.DATASET_DIR, exist_ok=True)

        self._refresh_plant_list()
        self.plant_detail_label.configure(text="")
        self.captured_images = []
        self.thumbnail_refs = []
        for child in self.thumb_strip.winfo_children():
            child.destroy()
        self.capture_progress_label.configure(text="0 images captured")
        self._set_status("All registered plants deleted", COLOR_SUBTEXT)

    # ====================================================================
    # MISC
    # ====================================================================
    def _set_status(self, text, color_hex):
        self.status_label.configure(text=text, foreground=color_hex)

    def _botanical_source_label(self):
        if self.botanical_classifier.is_available():
            if self.botanical_classifier.mode == "plantnet_api":
                return "Botanical model: Pl@ntNet API"
            if self.botanical_classifier.mode == "local_torchscript":
                return "Botanical model: local TorchScript"
        error = self.botanical_classifier.get_last_error()
        if error:
            return (
                f"Botanical model unavailable: {error}"
                " | Set PLANTNET_API_KEY in .env or configure the local TorchScript model."
            )
        return (
            "Botanical model unavailable: using registered plants fallback"
            " | Configure .env to enable Pl@ntNet or local TorchScript."
        )

    def on_close(self):
        self.camera.stop()
        self.root.destroy()


def run_app():
    root = tk.Tk()
    app = PlantRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
