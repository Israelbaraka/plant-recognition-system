# AI Plant Recognition System

A custom plant recognition app that works like a gallery-based face recognition system, but for the specific plants you register. It does **not** try to guess generic plant species. If a plant is not in the database, the app reports it as **Unknown Plant**.

## How it works

1. The camera captures a frame.
2. A green-plant detector finds the plant region.
3. A pretrained MobileNetV2 model extracts an embedding.
4. The recognition engine compares that embedding with stored embeddings in SQLite.
5. If the similarity is above the threshold, the plant is recognized. Otherwise, it is shown as unknown.

## Project layout

The source code lives in the `plant_recognition_system/` folder:

- `main.py` — application entry point
- `gui_app.py` — Tkinter GUI
- `camera_manager.py` — webcam and IP camera handling
- `plant_detector.py` — plant localization
- `feature_extractor.py` — embedding extraction
- `recognition_engine.py` — similarity matching
- `database.py` — SQLite storage layer
- `config.py` — app settings

## Requirements

- Python 3.9 to 3.12 recommended
- pip
- Internet access the first time you run the app, so PyTorch can download pretrained weights

## Install

From the project root:

```bash
cd plant_recognition_system
pip install -r requirements.txt
```

## Run

```bash
cd plant_recognition_system
python main.py
```

## Features

- Register new plants with multiple images
- Recognize plants in real time
- Use webcam, IP Webcam, DroidCam, or ESP32-CAM streams
- Store embeddings locally in SQLite
- Tune the match threshold from the GUI

## Notes

- The app creates its database automatically on first run.
- A one-time model download is required the first time the feature extractor runs.
- Local runtime data such as `plants.db`, `dataset/`, and virtual environments are ignored by Git.

## License

No license has been added yet.