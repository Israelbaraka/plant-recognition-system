# AI Plant Recognition System

An embedding-based plant recognition app that works like a gallery-driven face recognition system, but only for the plants you explicitly register. It does **not** try to guess generic plant species. If a plant is not in the database, the app returns **Unknown Plant**.

## Highlights

- Register new plants with multiple reference images
- Recognize plants in real time from a camera feed
- Support for webcam, IP Webcam, DroidCam, and ESP32-CAM streams
- Optional Pl@ntNet API mode for botanical species identification
- Automatic local registration when Pl@ntNet confidently detects a plant that is not yet in the gallery
- Local storage with SQLite and image embeddings
- Adjustable similarity threshold from the GUI

## How it works

1. The camera captures a frame.
2. A green-plant detector finds the likely plant region.
3. A pretrained MobileNetV2 model extracts an embedding.
4. The recognition engine compares that embedding with stored embeddings in SQLite.
5. If the similarity is above the threshold, the plant is recognized. Otherwise, it is shown as unknown.

When `BOTANICAL_RECOGNITION_MODE=plantnet_api` and `PLANTNET_API_KEY` is set, the app can also identify botanical species with Pl@ntNet. If the API returns a confident result for a plant that is not already registered locally, the app automatically saves the image and creates a new local plant entry.

## Project structure

The application code lives in `plant_recognition_system/`:

- `main.py` — application entry point
- `gui_app.py` — Tkinter GUI
- `camera_manager.py` — webcam and IP camera handling
- `plant_detector.py` — plant localization
- `feature_extractor.py` — embedding extraction
- `recognition_engine.py` — similarity matching
- `database.py` — SQLite storage layer
- `config.py` — app settings
- `README.md` — detailed project documentation in the subfolder

At the repository root you will also find:

- `README.md` — this overview file
- `.gitignore` — local files excluded from Git

## Requirements

- Python 3.9 to 3.12 recommended
- pip
- Internet access the first time you run the app, so PyTorch can download pretrained weights
- A Pl@ntNet API key is required only if you want botanical species identification and auto-registration

## Quick start

From the project root:

```bash
cd plant_recognition_system
pip install -r requirements.txt
python main.py
```

## Installation

If you prefer to run the steps separately:

```bash
cd plant_recognition_system
pip install -r requirements.txt
```

## Running the app

```bash
cd plant_recognition_system
python main.py
```

## Notes

- The app creates its database automatically on first run.
- A one-time model download is required the first time the feature extractor runs.
- Local runtime data such as `plants.db`, `dataset/`, and virtual environments are ignored by Git.
- Auto-registration from Pl@ntNet is controlled by `AUTO_REGISTER_PLANTNET_RESULTS` in `.env` and is enabled by default.

## License

No license has been added yet.