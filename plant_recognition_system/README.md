# AI Plant Recognition System

A custom, embedding-based plant recognition system — conceptually identical to
a face recognition system, except it recognizes **only the specific plants you
register**, not generic plant species. An unregistered plant is always
reported as **"Unknown Plant"**, never matched to the closest known one.

---

## 1. How It Works (Architecture)

```
                ┌────────────────────┐
                │   Camera Source     │   PC Webcam / IP Webcam / DroidCam / ESP32-CAM
                └─────────┬───────────┘
                          │ frame (BGR)
                          ▼
                ┌────────────────────┐
                │   Plant Detector    │   HSV + contour heuristic -> bounding box
                └─────────┬───────────┘
                          │ cropped plant region
                          ▼
                ┌────────────────────┐
                │  Feature Extractor  │   Pretrained MobileNetV2 (PyTorch)
                │  (Embedding model)  │   -> 1280-D L2-normalized vector
                └─────────┬───────────┘
                          │ embedding
           ┌──────────────┴───────────────┐
           ▼                               ▼
  REGISTRATION MODE                RECOGNITION MODE
  Save image + embedding           Compare against ALL stored
  to dataset/ and SQLite           embeddings (cosine similarity)
                                   -> best match above threshold
                                      = Recognized
                                   -> below threshold
                                      = Unknown Plant
```

This mirrors how face recognition systems work: instead of classifying into a
fixed set of species (which would require huge labeled datasets and
retraining for every new plant), the system stores an **embedding gallery**
per registered plant and does similarity search at inference time. Adding a
new plant is just adding new gallery entries — no retraining required, and
existing plants are completely unaffected.

### Why this guarantees "Unknown" stays Unknown
The recognition engine (`recognition_engine.py`) applies a **hard similarity
threshold**. Even if a query image is *closer* to "Rose" than to anything
else in the gallery, if that similarity score is still below the threshold,
it is reported as "Unknown Plant" — never silently assigned to the nearest
registered plant. You can tune this threshold live from the GUI slider.

---

## 2. Project Structure

```
plant_recognition_system/
├── main.py                 # Entry point — run this
├── config.py                # All tunable settings (paths, thresholds, camera defaults)
├── database.py               # SQLite layer (plants + embeddings tables)
├── feature_extractor.py      # CNN embedding model (MobileNetV2 backbone)
├── plant_detector.py         # HSV/contour-based bounding box localizer
├── recognition_engine.py     # Cosine-similarity matching + threshold logic
├── camera_manager.py         # Unified webcam / IP Webcam / DroidCam / ESP32-CAM handler
├── gui_app.py                 # Tkinter GUI (all the buttons/panels described below)
├── requirements.txt
├── plants.db                  # SQLite database (created automatically on first run)
└── dataset/
    ├── Rose/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Mango/
    │   └── ...
    └── Aloe Vera/
        └── ...
```

---

## 3. Installation Guide

### Step 1 — Install Python
Python 3.9–3.12 recommended.

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> The first time you run the app, PyTorch will automatically download the
> pretrained MobileNetV2 weights (~14 MB) from `download.pytorch.org`. This
> requires an internet connection **once** — after that, everything runs
> fully offline, including all recognition (no internet needed for actual
> use, only for that initial model download).

### Step 4 — Run the application
```bash
python main.py
```

---

## 4. Connecting Each Camera Type

The GUI has a **Camera Source** dropdown with 4 options. Pick one, fill in
the **Index / URL** field, then click **Start Camera**.

| Source | What to enter | Notes |
|---|---|---|
| `webcam` | An integer, e.g. `0` | `0` is usually the default built-in/USB webcam. Try `1`, `2`, etc. if you have multiple. |
| `ip_webcam` | `http://<phone-ip>:8080/video` | Install the **IP Webcam** app (Android), start the server, use the URL it displays. Phone and PC must be on the same Wi-Fi network. |
| `droidcam` | `http://<phone-ip>:4747/video` | Install **DroidCam**, start it, use the IP shown in the app. |
| `esp32cam` | `http://<esp32-ip>/stream` | Flash the standard ESP32-CAM "CameraWebServer" example sketch; the device prints its IP address to Serial on boot. |

All three stream-based sources are handled identically by OpenCV
(`cv2.VideoCapture(url)`), so no source-specific code is needed beyond the
URL.

---

## 5. Using the App

### Registration Mode (teaching the system a new plant)
1. Start a camera source.
2. Type a unique **Plant Name** (e.g. `Aloe Vera`).
3. Point the camera at the plant and click **Capture Images**.
   - The app automatically captures 15 images (configurable via
     `IMAGES_PER_REGISTRATION` in `config.py`), one every ~350ms.
   - Slowly rotate the plant / move the camera around it between shots so
     you capture different angles — this is what makes the embedding
     gallery robust, just like enrolling several face photos.
   - Thumbnails of captured images appear in the preview strip.
   - You can click **Capture Images** again to add more shots before saving.
4. Click **Register Plant**.
   - Images are saved to `dataset/<PlantName>/imageN.jpg`.
   - An embedding is computed for each image and stored in `plants.db`.
   - The plant immediately appears in the **Registered Plants** list and is
     available for recognition right away — no restart needed.

### Recognition Mode (finding registered plants in real time)
1. Start a camera source.
2. Click **Start Recognition**.
3. The app continuously scans frames (every 5th frame by default, for
   performance — see `RECOGNITION_FRAME_SKIP` in `config.py`):
   - **Green box + plant name + confidence%** → match found above threshold.
   - **Red box + "Unknown Plant" + confidence%** → no registered plant
     matched closely enough.
4. Adjust the **Match Threshold** slider live if you get too many false
   positives (lower confidence floor lets in bad matches — raise the
   threshold) or too many false "Unknowns" for plants you did register
   (lower the threshold slightly).

### Deleting a Plant
Select it in the **Registered Plants** list and click **Delete Selected
Plant**. This removes its database rows (embeddings + plant record) and its
`dataset/<PlantName>/` folder. All other plants are completely unaffected,
and recognition immediately stops considering it a possible match.

---

## 6. Database Schema

`plants.db` (SQLite):

```sql
CREATE TABLE plants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    registered_at TEXT NOT NULL,     -- ISO timestamp
    num_images INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    vector BLOB NOT NULL,             -- float32 numpy array, serialized
    FOREIGN KEY (plant_id) REFERENCES plants(id) ON DELETE CASCADE
);
```

Every registered image gets its own embedding row. Recognition compares a
new frame against **every stored embedding** (not just one average per
plant) and uses the top-k closest matches — this is the same gallery-search
approach used in real face recognition systems, and it tolerates lighting
/ angle variation much better than a single average vector per plant.

---

## 7. Key Design Decisions & Honest Limitations

- **Embedding model**: We use an ImageNet-pretrained MobileNetV2 (general
  visual feature extractor), not a model trained specifically for plant
  re-identification. This works well for distinguishing visually distinct
  registered plants and is fast enough for real-time CPU inference, but it
  is not as precision-tuned as a purpose-trained metric-learning model
  (like FaceNet is for faces). If you need higher accuracy:
  - Increase `IMAGES_PER_REGISTRATION` for a richer gallery per plant.
  - Fine-tune the backbone on your own plant photos with a triplet/contrastive
    loss (swap into `feature_extractor.py` — the rest of the pipeline doesn't
    need to change, since it only depends on `extract()` returning a
    normalized vector).
- **Plant localization**: `plant_detector.py` uses HSV-based green
  segmentation + contours, not a trained object detector. It works well for
  a single plant roughly centered in frame (desk/windowsill setup). For
  tighter or multi-plant detection, replace it with a fine-tuned YOLOv8
  model — `detect()` just needs to keep returning `(x, y, w, h)` or `None`.
- **No internet required at runtime**: after the one-time pretrained-weight
  download, everything (camera capture, feature extraction, database,
  matching) runs 100% locally.

---

## 8. Tuning Cheat Sheet (`config.py`)

| Setting | Effect |
|---|---|
| `SIMILARITY_THRESHOLD` | Higher = stricter matching (fewer false positives, more "Unknown"s for genuine plants). Lower = looser (more matches, more risk of false positives). |
| `IMAGES_PER_REGISTRATION` | More images per plant = more robust gallery, slower registration. |
| `TOP_K_MATCHES` | How many nearest gallery entries are averaged for the confidence score. |
| `RECOGNITION_FRAME_SKIP` | Higher = better GUI performance, slightly less "live" feeling recognition. |
| `MIN_CONTOUR_AREA_RATIO` | Minimum size (as a fraction of the frame) a green blob must be to count as a detected plant. |
