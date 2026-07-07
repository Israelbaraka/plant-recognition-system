"""
botanical_classifier.py
-----------------------
Botanical species classifier used before the original registered-plant
gallery matcher.

Two botanical backends are supported:
    1. Pl@ntNet API, for real species identification from plant images.
    2. A local TorchScript classifier, for offline botanical models.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config

try:
    import requests
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    requests = None


@dataclass
class BotanicalPrediction:
    name: str
    confidence: float
    source: str
    scientific_name: Optional[str] = None
    family: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_confident(self) -> bool:
        return (self.confidence / 100.0) >= config.BOTANICAL_CONFIDENCE_THRESHOLD


class BotanicalClassifier:
    def __init__(self, mode: str = None, device: str = None):
        self.mode = mode or config.BOTANICAL_RECOGNITION_MODE
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._local_model = None
        self._local_labels = None
        self._last_error = None

        self.preprocess = transforms.Compose([
            transforms.Resize((config.BOTANICAL_INPUT_SIZE, config.BOTANICAL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    def get_last_error(self) -> Optional[str]:
        return self._last_error

    def is_available(self) -> bool:
        if self.mode == "plantnet_api":
            return bool(config.PLANTNET_API_KEY) and requests is not None
        if self.mode == "local_torchscript":
            return os.path.isfile(config.BOTANICAL_MODEL_PATH) and os.path.isfile(config.BOTANICAL_LABELS_PATH)
        return False

    def classify(self, image_bgr: np.ndarray) -> Optional[BotanicalPrediction]:
        self._last_error = None

        if image_bgr is None or image_bgr.size == 0:
            self._last_error = "Empty plant crop."
            return None

        if self.mode == "plantnet_api":
            return self._classify_with_plantnet(image_bgr)
        if self.mode == "local_torchscript":
            return self._classify_with_local_model(image_bgr)

        self._last_error = f"Unsupported botanical mode: {self.mode}"
        return None

    def _classify_with_plantnet(self, image_bgr: np.ndarray) -> Optional[BotanicalPrediction]:
        if requests is None:
            self._last_error = "The 'requests' package is required for Pl@ntNet API mode."
            return None
        if not config.PLANTNET_API_KEY:
            self._last_error = "Set PLANTNET_API_KEY to enable botanical species recognition."
            return None

        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            self._last_error = "Could not encode plant crop as JPEG."
            return None

        url = f"{config.PLANTNET_ENDPOINT}/{config.PLANTNET_PROJECT}"
        params = {"api-key": config.PLANTNET_API_KEY}
        files = [("images", ("plant.jpg", encoded.tobytes(), "image/jpeg"))]
        data = {"organs": config.PLANTNET_ORGAN}

        try:
            response = requests.post(
                url,
                params=params,
                files=files,
                data=data,
                timeout=config.PLANTNET_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            self._last_error = f"Pl@ntNet request failed: {exc}"
            return None
        except json.JSONDecodeError:
            self._last_error = "Pl@ntNet returned an invalid JSON response."
            return None

        results = payload.get("results") or []
        if not results:
            self._last_error = "No botanical species found by Pl@ntNet."
            return None

        best = results[0]
        species = best.get("species") or {}
        scientific_name = species.get("scientificNameWithoutAuthor") or species.get("scientificName")
        common_names = species.get("commonNames") or []
        family = (species.get("family") or {}).get("scientificNameWithoutAuthor")
        display_name = common_names[0] if common_names else scientific_name or "Unknown species"
        score = float(best.get("score") or 0.0)

        return BotanicalPrediction(
            name=display_name,
            scientific_name=scientific_name,
            family=family,
            confidence=max(0.0, min(100.0, score * 100.0)),
            source="Pl@ntNet",
        )

    def _classify_with_local_model(self, image_bgr: np.ndarray) -> Optional[BotanicalPrediction]:
        try:
            model = self._load_local_model()
            labels = self._load_local_labels()
        except (OSError, RuntimeError, ValueError) as exc:
            self._last_error = str(exc)
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence, index = torch.max(probs, dim=0)

        label_index = int(index.item())
        if label_index >= len(labels):
            self._last_error = "Local model predicted a label index not present in labels file."
            return None

        return BotanicalPrediction(
            name=labels[label_index],
            scientific_name=labels[label_index],
            confidence=float(confidence.item() * 100.0),
            source="Local botanical model",
        )

    def _load_local_model(self):
        if self._local_model is None:
            if not os.path.isfile(config.BOTANICAL_MODEL_PATH):
                raise OSError(f"Local botanical model not found: {config.BOTANICAL_MODEL_PATH}")
            model = torch.jit.load(config.BOTANICAL_MODEL_PATH, map_location=self.device)
            model.eval()
            self._local_model = model
        return self._local_model

    def _load_local_labels(self):
        if self._local_labels is None:
            if not os.path.isfile(config.BOTANICAL_LABELS_PATH):
                raise OSError(f"Botanical labels file not found: {config.BOTANICAL_LABELS_PATH}")
            with open(config.BOTANICAL_LABELS_PATH, "r", encoding="utf-8") as handle:
                labels = [line.strip() for line in handle if line.strip()]
            if not labels:
                raise ValueError("Botanical labels file is empty.")
            self._local_labels = labels
        return self._local_labels
