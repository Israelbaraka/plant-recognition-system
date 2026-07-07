"""
feature_extractor.py
---------------------
Turns a plant image (BGR numpy array from OpenCV) into a fixed-length,
L2-normalized embedding vector -- the same core idea used in face
recognition systems (FaceNet/ArcFace style), just applied to plants.

We use an ImageNet-pretrained MobileNetV2 with the classification head
removed, and global-average-pool the final feature maps. This is NOT
trained specifically to discriminate between plant species/instances,
but pretrained CNN features are well known to transfer well to "is
this the same object" style similarity tasks, which is exactly what
we need here: not "what species is this", but "have I seen this exact
registered plant before".

If you want higher accuracy down the line, see the README section on
fine-tuning with a triplet/contrastive loss on your own plant photos.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

import config


class FeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained MobileNetV2 and strip the classifier head so
        # the forward pass returns raw feature maps instead of 1000
        # ImageNet class logits.
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # convolutional feature extractor only
        self.features.eval()
        self.features.to(self.device)

        # Standard ImageNet preprocessing (the pretrained weights expect this).
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.EMBEDDING_INPUT_SIZE, config.EMBEDDING_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        # Disable gradient tracking globally for this module -- we only
        # ever run inference, never train the backbone.
        for p in self.features.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            image_bgr: HxWx3 uint8 image straight from OpenCV (BGR order).
        Returns:
            1D float32 numpy array of length config.EMBEDDING_DIM,
            L2-normalized (so cosine similarity == dot product).
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot extract features from an empty image.")

        # OpenCV gives BGR; torchvision transforms expect RGB.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        feature_maps = self.features(tensor)               # (1, 1280, h, w)
        pooled = nn.functional.adaptive_avg_pool2d(feature_maps, 1)  # (1, 1280, 1, 1)
        embedding = pooled.flatten(1).squeeze(0)            # (1280,)

        embedding = embedding.cpu().numpy().astype(np.float32)

        # L2 normalize so that cosine similarity = simple dot product downstream.
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding
