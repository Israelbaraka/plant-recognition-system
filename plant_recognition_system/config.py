"""
config.py
---------
Central configuration for the AI Plant Recognition System.
Keep all tunable constants here so the rest of the codebase never
hardcodes paths, thresholds, or camera settings.
"""

import os

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")  # dataset/<PlantName>/imageN.jpg
DB_PATH = os.path.join(BASE_DIR, "plants.db")  # SQLite database file

os.makedirs(DATASET_DIR, exist_ok=True)


def _load_local_env_file():
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    candidate_paths = [
        os.path.join(os.path.dirname(BASE_DIR), ".env"),
        os.path.join(BASE_DIR, ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.isfile(env_path):
            continue

        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_local_env_file()


def _resolve_botanical_mode() -> str:
    """Pick the best botanical mode based on available configuration."""
    requested_mode = os.getenv("BOTANICAL_RECOGNITION_MODE", "auto").strip().lower()

    if requested_mode in {"plantnet_api", "local_torchscript", "registered_gallery"}:
        return requested_mode

    if os.getenv("PLANTNET_API_KEY", "").strip():
        return "plantnet_api"

    model_path = os.getenv(
        "BOTANICAL_MODEL_PATH",
        os.path.join(BASE_DIR, "models", "botanical_classifier.pt"),
    )
    labels_path = os.getenv(
        "BOTANICAL_LABELS_PATH",
        os.path.join(BASE_DIR, "models", "botanical_labels.txt"),
    )
    if os.path.isfile(model_path) and os.path.isfile(labels_path):
        return "local_torchscript"

    return "registered_gallery"


# ----------------------------------------------------------------------
# REGISTRATION SETTINGS
# ----------------------------------------------------------------------
# Number of images captured per plant during registration. More images
# from different angles -> a richer "gallery" for matching -> more
# robust recognition (exactly like enrolling multiple face photos).
IMAGES_PER_REGISTRATION = 15

# Minimum images required before a plant is allowed to be saved.
MIN_IMAGES_TO_REGISTER = 5

# Delay (in milliseconds) between automatic captures during a capture burst.
CAPTURE_INTERVAL_MS = 350

# ----------------------------------------------------------------------
# RECOGNITION SETTINGS
# ----------------------------------------------------------------------
# Embedding vectors are compared with cosine similarity, which ranges
# from -1 (opposite) to 1 (identical direction). For L2-normalized
# CNN feature vectors, real-world "same object" matches typically sit
# well above 0.6-0.7, while unrelated objects fall lower.
#
# IMPORTANT: This threshold is the single most important tuning knob
# in the whole system. Too low -> false positives (unknown plants get
# misidentified). Too high -> real matches get rejected as "Unknown".
# Adjust this value from the GUI's threshold slider as you test with
# your own plants and lighting conditions.
SIMILARITY_THRESHOLD = 0.72

# Botanical recognition mode:
#   - "plantnet_api": identifies botanical species with the Pl@ntNet API.
#   - "local_torchscript": uses a local TorchScript plant classifier.
#   - "registered_gallery": uses the original custom registered-plant gallery.
#   - "auto": selects PlantNet if a key is available, otherwise local
#     TorchScript if both files exist, otherwise registered gallery.
#
# Pl@ntNet needs an API key. Local TorchScript mode needs a model checkpoint
# and labels file configured below.
BOTANICAL_RECOGNITION_MODE = _resolve_botanical_mode()
BOTANICAL_CONFIDENCE_THRESHOLD = 0.40

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "")
PLANTNET_PROJECT = os.getenv("PLANTNET_PROJECT", "all")
PLANTNET_ORGAN = os.getenv("PLANTNET_ORGAN", "leaf")
PLANTNET_ENDPOINT = "https://my-api.plantnet.org/v2/identify"
PLANTNET_TIMEOUT_SECONDS = 12

BOTANICAL_MODEL_PATH = os.getenv(
    "BOTANICAL_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "botanical_classifier.pt"),
)
BOTANICAL_LABELS_PATH = os.getenv(
    "BOTANICAL_LABELS_PATH",
    os.path.join(BASE_DIR, "models", "botanical_labels.txt"),
)
BOTANICAL_INPUT_SIZE = 224

# How many of the closest gallery embeddings to average when scoring
# a candidate match (k-NN style voting). 1 = nearest neighbor only.
TOP_K_MATCHES = 3

# Run recognition on every Nth frame to keep the GUI responsive
# (full embedding extraction is the expensive step).
RECOGNITION_FRAME_SKIP = 8

# Downscale frames during recognition to reduce CPU work while keeping
# the UI responsive. 1.0 = full size, 0.75 = 75% size, etc.
RECOGNITION_ANALYSIS_SCALE = 0.65

# ----------------------------------------------------------------------
# FEATURE EXTRACTOR (EMBEDDING MODEL)
# ----------------------------------------------------------------------
# Backbone CNN used to turn a plant image crop into a fixed-length
# feature vector ("embedding"). We use an ImageNet-pretrained
# MobileNetV2 with its classification head removed, which is fast
# enough for real-time use on CPU and produces a 1280-D embedding
# that captures texture, color, and shape -- enough to discriminate
# between distinct plants the user has registered.
EMBEDDING_DIM = 1280
EMBEDDING_INPUT_SIZE = 224  # Standard ImageNet input resolution

# ----------------------------------------------------------------------
# CAMERA SETTINGS
# ----------------------------------------------------------------------
# Camera "source" can be:
#   - an integer index for a local webcam (e.g. 0, 1)
#   - a URL string for IP Webcam / DroidCam / ESP32-CAM MJPEG streams
#
# Common examples (edit per device in the GUI, these are just defaults):
DEFAULT_WEBCAM_INDEX = 0
DEFAULT_IP_WEBCAM_URL = "http://192.168.1.50:8080/video"  # IP Webcam (Android)
DEFAULT_DROIDCAM_URL = "http://192.168.1.50:4747/video"  # DroidCam
DEFAULT_ESP32CAM_URL = "http://192.168.1.60/stream"  # ESP32-CAM MJPEG stream

CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480

# GUI video preview refresh rate (milliseconds between frame redraws)
PREVIEW_REFRESH_MS = 30

# Minimum fraction of pixels that must look plant-like (green-ish) before
# the system will consider a frame for capture/recognition.
MIN_GREEN_PIXEL_RATIO = 0.015

# ----------------------------------------------------------------------
# PLANT DETECTOR (bounding box heuristic)
# ----------------------------------------------------------------------
# We use an HSV color-based + contour heuristic to localize the most
# "plant-like" (green / leafy) region in the frame for the bounding
# box. It is not a trained object detector -- see README for how to
# swap in a YOLO model later if you need tighter, multi-plant boxes.
MIN_CONTOUR_AREA_RATIO = 0.02  # contour must cover >=2% of frame area
