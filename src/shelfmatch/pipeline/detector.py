"""
Detection module using Grounding DINO and OWLv2.

Supports:
- Text-prompted zero-shot detection (Grounding DINO)
- Image-prompted zero-shot detection (OWLv2)
- Confidence/threshold tuning
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection from a detector."""
    bbox: np.ndarray   # [x1, y1, x2, y2] normalized 0-1
    score: float
    label: str
    source: str  # "grounding_dino" | "owlv2"


@dataclass
class DetectionOutput:
    """All detections from a single image."""
    image_size: tuple[int, int]  # (width, height)
    detections: list[DetectionResult] = field(default_factory=list)


class GroundingDINODetector:
    """Zero-shot detection via Grounding DINO + text prompts."""

    MODEL_ID = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        logger.info("Loading Grounding DINO: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompts: list[str],
    ) -> DetectionOutput:
        """
        Detect objects in image using text prompts.

        Args:
            image: PIL Image
            text_prompts: List of text descriptions, e.g. ["product A", "product B"]

        Returns:
            DetectionOutput with all detections
        """
        self._load_model()
        w, h = image.size
        inputs = self._processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=[(h, w)],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )[0]

        detections = []
        for score, label, bbox in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append(
                DetectionResult(
                    bbox=bbox.cpu().numpy() / np.array([w, h, w, h]),
                    score=score.item(),
                    label=label,
                    source="grounding_dino",
                )
            )

        logger.info(
            "Grounding DINO detected %d objects from %d prompts",
            len(detections),
            len(text_prompts),
        )
        return DetectionOutput(image_size=(w, h), detections=detections)

    def set_thresholds(self, box: float, text: float):
        self.box_threshold = box
        self.text_threshold = text


class OWLv2Detector:
    """Zero-shot detection via OWLv2 — image-prompt or text-prompt."""

    MODEL_ID = "google/owlv2-base-patch16-ensemble"

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: Optional[str] = None,
        score_threshold: float = 0.3,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        logger.info("Loading OWLv2: %s", self.model_id)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompts: Optional[list[str]] = None,
        reference_images: Optional[list[Image.Image]] = None,
    ) -> DetectionOutput:
        """
        Detect objects using text prompts or reference image crops.

        Args:
            image: PIL Image (the shelf image)
            text_prompts: List of text labels
            reference_images: List of reference product images (image-prompt mode)
        """
        self._load_model()
        w, h = image.size

        if text_prompts:
            inputs = self._processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
            )
        elif reference_images:
            inputs = self._processor(
                query_images=reference_images,
                images=image,
                return_tensors="pt",
            )
        else:
            raise ValueError("Must provide either text_prompts or reference_images")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]

        detections = []
        label_key = "labels" if text_prompts else "query_labels"
        for score, label_id, bbox in zip(
            results["scores"], results[label_key], results["boxes"]
        ):
            label_str = (
                text_prompts[label_id]
                if text_prompts
                else f"ref_{label_id}"
            )
            detections.append(
                DetectionResult(
                    bbox=bbox.cpu().numpy() / np.array([w, h, w, h]),
                    score=score.item(),
                    label=label_str,
                    source="owlv2",
                )
            )

        logger.info("OWLv2 detected %d objects", len(detections))
        return DetectionOutput(image_size=(w, h), detections=detections)


class YOLOWorldDetector:
    """
    YOLO-World for fast real-time detection.
    Fine-tunable for product-specific detection.

    Note: YOLO-World typically runs via ultralytics API.
    This class wraps it for the unified detector interface.
    """

    def __init__(
        self,
        model_size: str = "l",  # s, m, l
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
    ):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self._model = None
        self._model_id = f"yolov8{model_size}-world"

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLOWorld
        except ImportError as e:
            raise RuntimeError(
                "YOLO-World requires `ultralytics`. Install with: pip install ultralytics"
            ) from e

        logger.info("Loading YOLO-World-%s", self.model_size)
        self._model = YOLOWorld(f"{self._model_id}.pt")
        self._model.to(self.device)

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        text_prompts: list[str],
    ) -> DetectionOutput:
        """
        Detect using YOLO-World with text-prompted classes.
        """
        self._load_model()
        w, h = image.size

        # Set text prompts as detection classes
        self._model.set_classes(text_prompts)

        results = self._model.predict(
            image,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append(
                    DetectionResult(
                        bbox=np.array([x1, y1, x2, y2]) / np.array([w, h, w, h]),
                        score=conf,
                        label=text_prompts[cls_id],
                        source="yoloworld",
                    )
                )

        logger.info("YOLO-World detected %d objects", len(detections))
        return DetectionOutput(image_size=(w, h), detections=detections)
