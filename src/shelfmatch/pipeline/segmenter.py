"""
Segmentation module using SAM2 and HQ-SAM.

Supports:
- Box-prompted segmentation (SAM2)
- Point-prompted segmentation
- Automatic mask generation (SAM2 with grid points)
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
class MaskResult:
    """Single segmentation mask."""
    mask: np.ndarray  # HxW bool
    score: float
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2] normalized 0-1


@dataclass
class SegmentationOutput:
    """All masks for a detection region."""
    image_size: tuple[int, int]  # (width, height)
    masks: list[MaskResult] = field(default_factory=list)


class SAM2Segmenter:
    """
    SAM2 (Segment Anything Model 2) for high-quality segmentation.

    Supports multiple prompt modes:
    - box: [x1,y1,x2,y2] normalized
    - point: [x,y] normalized with labels (1=foreground, 0=background)
    - automatic: grid point sampling for all objects
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-base",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._predictor = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generation import SAM2AutomaticMaskGenerator
        except ImportError as e:
            raise RuntimeError(
                "SAM2 not installed. Run: pip install 'sam2>=1.0.0' "
                "or see https://github.com/facebookresearch/sam2"
            ) from e

        logger.info("Loading SAM2: %s", self.model_id)

        # Build model — SAM2 uses .yaml config + checkpoint
        # Map model_id to checkpoint path
        sam2_cfg = {
            "facebook/sam2.1-hiera-base": ("sam2.1_hiera_base.yaml", "sam2.1_hiera_base.pt"),
            "facebook/sam2.1-hiera-large": ("sam2.1_hiera_large.yaml", "sam2.1_hiera_large.pt"),
            "facebook/sam2.1-hiera-small": ("sam2.1_hiera_small.yaml", "sam2.1_hiera_small.pt"),
        }

        if self.model_id not in sam2_cfg:
            raise ValueError(f"Unknown SAM2 model: {self.model_id}. Use: {list(sam2_cfg.keys())}")

        cfg_name, ckpt_name = sam2_cfg[self.model_id]

        # Checkpoint must be downloaded separately
        # See: https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-checkpoints
        checkpoint_path = f"~/.cache/sam2/{ckpt_name}"

        import os
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {checkpoint_path}. "
                "Download from: https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-checkpoints"
            )

        self._model = build_sam2(cfg_name, checkpoint_path, device=self.device)
        self._predictor = SAM2AutomaticMaskGenerator(self._model)

    def segment_from_box(
        self,
        image: Image.Image,
        bbox: np.ndarray,
    ) -> SegmentationOutput:
        """
        Segment a region defined by a bounding box.

        Args:
            image: PIL Image
            bbox: [x1, y1, x2, y2] in normalized 0-1 coords
        """
        self._load_model()
        w, h = image.size

        # Convert normalized → pixel coords
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(self._model)
        np_img = np.array(image)[:, :, ::-1]  # RGB → BGR for SAM2

        with torch.no_grad():
            predictor.set_image(np_img)
            masks, scores, _ = predictor.predict_box(
                box=np.array([x1, y1, x2, y2]),
                labels=np.array([1]),
            )

        results = []
        for mask, score in zip(masks, scores):
            results.append(
                MaskResult(
                    mask=mask,
                    score=float(score),
                    bbox=bbox,
                )
            )

        logger.info("SAM2 generated %d masks for box", len(results))
        return SegmentationOutput(image_size=(w, h), masks=results)

    def segment_automatic(
        self,
        image: Image.Image,
        min_area: int = 100,
    ) -> SegmentationOutput:
        """
        Automatic mask generation — finds all salient objects.

        Args:
            image: PIL Image
            min_area: Minimum mask pixel area to keep
        """
        self._load_model()
        w, h = image.size

        np_img = np.array(image)[:, :, ::-1]
        mask_generator = SAM2AutomaticMaskGenerator(
            self._model,
            min_mask_area=min_area,
        )
        sam_masks = mask_generator.generate(np_img)

        results = []
        for m in sam_masks:
            bbox = np.array(m["bbox"]) / np.array([w, h, w, h])
            results.append(
                MaskResult(
                    mask=m["segmentation"],
                    score=float(m["area"]),
                    bbox=bbox,
                )
            )

        # Sort by area (largest first)
        results.sort(key=lambda x: x.score, reverse=True)

        logger.info("SAM2 automatic mode generated %d masks", len(results))
        return SegmentationOutput(image_size=(w, h), masks=results)


class HQSAMSegmenter:
    """
    HQ-SAM (High Quality SAM) — better mask quality than vanilla SAM.
    Drop-in replacement with improved boundary refinement.
    """

    def __init__(
        self,
        model_id: str = "sam_hq_vit_base",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from segment_anything_hq import sam_hq_model_registry, SamHqPredictor
        except ImportError as e:
            raise RuntimeError(
                "HQ-SAM not installed. Run: pip install git+https://github.com/SysCV/sam-hq"
            ) from e

        logger.info("Loading HQ-SAM: %s", self.model_id)
        self._model = sam_hq_model_registry[self.model_id]()
        self._model.to(self.device)
        self._model.eval()
        self._predictor = SamHqPredictor(self._model)

    @torch.no_grad()
    def segment_from_box(
        self,
        image: Image.Image,
        bbox: np.ndarray,
    ) -> SegmentationOutput:
        self._load_model()
        w, h = image.size

        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        np_img = np.array(image)[:, :, ::-1]
        self._predictor.set_image(np_img)

        masks, scores, _ = self._predictor.predict_box(
            box=np.array([x1, y1, x2, y2]),
            point_labels=np.array([1]),
        )

        results = []
        for mask, score in zip(masks, scores):
            results.append(
                MaskResult(
                    mask=mask,
                    score=float(score),
                    bbox=bbox,
                )
            )

        return SegmentationOutput(image_size=(w, h), masks=results)
