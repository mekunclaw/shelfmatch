"""
ShelfMatcher — unified pipeline for reference-based shelf product detection.

Architecture:
  Reference Images ──→ FeatureExtractor → FAISS index
                                          ↓
  Shelf Image ──→ GroundingDINO (detect) ──→ SAM2 (segment)
                                          ↓
                                   Crop regions
                                          ↓
                              FeatureExtractor (crop)
                                          ↓
                              FAISS similarity search
                                          ↓
                              MatchResult: bbox + mask + product_id
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass, field
import logging
from pathlib import Path

from shelfmatch.pipeline.detector import (
    DetectionResult,
    DetectionOutput,
    GroundingDINODetector,
    OWLv2Detector,
    YOLOWorldDetector,
)
from shelfmatch.pipeline.segmenter import SAM2Segmenter, MaskResult
from shelfmatch.pipeline.matcher import (
    ReferenceProduct,
    MatchResult,
    ProductMatcher,
    FeatureExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class ShelfMatchResult:
    """Complete result for one shelf image."""
    shelf_image_path: str
    matches: list[MatchResult]
    num_references: int
    pipeline_version: str = "0.1.0"


@dataclass
class PipelineConfig:
    """Configuration for the detection pipeline."""
    # Detector
    detector_type: str = "grounding_dino"  # "grounding_dino" | "owlv2" | "yoloworld"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    # Segmenter
    segmenter_type: str = "sam2"  # "sam2" | "hqsam"
    min_mask_area: int = 100
    # Matcher
    feature_model: str = "siglip"  # "siglip" | "dinov2"
    similarity_threshold_high: float = 0.85
    similarity_threshold_medium: float = 0.70
    # Pipeline options
    run_segmentation: bool = True
    use_multi_angle: bool = True


class ShelfMatcher:
    """
    Unified pipeline for reference-based shelf product detection.

    Usage:
        matcher = ShelfMatcher()
        matcher.load_references(["product1.jpg", "product2.jpg"])
        results = matcher.detect("shelf.jpg")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._detector: Optional[GroundingDINODetector | OWLv2Detector | YOLOWorldDetector] = None
        self._segmenter: Optional[SAM2Segmenter] = None
        self._matcher: Optional[ProductMatcher] = None
        self._reference_products: list[ReferenceProduct] = []
        self._reference_names: list[str] = []

    # ─── Detectors ───────────────────────────────────────────────

    @property
    def detector(self):
        if self._detector is None:
            dt = self.config.detector_type
            if dt == "grounding_dino":
                self._detector = GroundingDINODetector(
                    box_threshold=self.config.box_threshold,
                    text_threshold=self.config.text_threshold,
                )
            elif dt == "owlv2":
                self._detector = OWLv2Detector(
                    score_threshold=self.config.box_threshold,
                )
            elif dt == "yoloworld":
                self._detector = YOLOWorldDetector(
                    conf_threshold=self.config.box_threshold,
                )
            else:
                raise ValueError(f"Unknown detector: {dt}")
            logger.info("Initialized %s detector", dt)
        return self._detector

    @property
    def segmenter(self) -> SAM2Segmenter:
        if self._segmenter is None:
            self._segmenter = SAM2Segmenter()
            logger.info("Initialized SAM2 segmenter")
        return self._segmenter

    @property
    def matcher(self) -> ProductMatcher:
        if self._matcher is None:
            extractor = FeatureExtractor(model_type=self.config.feature_model)
            self._matcher = ProductMatcher(
                feature_extractor=extractor,
                similarity_threshold_high=self.config.similarity_threshold_high,
                similarity_threshold_medium=self.config.similarity_threshold_medium,
            )
            logger.info("Initialized ProductMatcher with %s", self.config.feature_model)
        return self._matcher

    # ─── Reference Management ─────────────────────────────────────

    def load_references(
        self,
        reference_paths: list[str | Path],
        product_names: Optional[list[str]] = None,
    ):
        """
        Load reference product images and build the FAISS index.

        Args:
            reference_paths: Paths to reference images (one per product)
            product_names: Optional names for each product. Defaults to filename stems.
        """
        from shelfmatch.pipeline.matcher import FeatureExtractor

        paths = [Path(p) for p in reference_paths]
        names = product_names or [p.stem for p in paths]

        logger.info("Loading %d reference products", len(paths))

        # Extract features from each reference
        extractor = FeatureExtractor(model_type=self.config.feature_model)

        self._reference_products = []
        self._reference_names = names

        for i, (path, name) in enumerate(zip(paths, names)):
            img = Image.open(path).convert("RGB")
            feat = extractor.extract([img])[0]

            product = ReferenceProduct(
                product_id=f"product_{i:03d}",
                name=name,
                features=feat,
                reference_image=img,
            )
            self._reference_products.append(product)

        # Build FAISS index
        self.matcher.build_index(self._reference_products)
        logger.info("Reference index built with %d products", len(self._reference_products))

    def load_references_multi_angle(
        self,
        reference_groups: list[list[str | Path]],
        product_names: Optional[list[str]] = None,
    ):
        """
        Load reference products with multiple angles each.

        Args:
            reference_groups: List of groups, where each group is a list of
                              paths for different angles of the same product.
            product_names: Optional names for each product group.
        """
        from shelfmatch.pipeline.matcher import FeatureExtractor

        names = product_names or [f"product_{i}" for i in range(len(reference_groups))]
        extractor = FeatureExtractor(model_type=self.config.feature_model)

        self._reference_products = []
        self._reference_names = names

        for i, (group, name) in enumerate(zip(reference_groups, names)):
            paths = [Path(p) for p in group]
            images = [Image.open(p).convert("RGB") for p in paths]

            # Extract features for each angle
            angle_feats = extractor.extract(images)

            # Use first image as primary feature, rest as aux
            primary_feat = angle_feats[0]

            product = ReferenceProduct(
                product_id=f"product_{i:03d}",
                name=name,
                features=primary_feat,
                reference_image=images[0],
                multi_angle_features=angle_feats[1:],
            )
            self._reference_products.append(product)

        self.matcher.build_index(self._reference_products)
        logger.info(
            "Multi-angle reference index built with %d products (%d total angles)",
            len(self._reference_products),
            sum(len(g) for g in reference_groups),
        )

    # ─── Main Pipeline ────────────────────────────────────────────

    def detect(
        self,
        shelf_image: str | Path | Image.Image,
        text_prompts: Optional[list[str]] = None,
        return_masks: bool = True,
    ) -> ShelfMatchResult:
        """
        Run the full detection pipeline on a shelf image.

        Args:
            shelf_image: Path or PIL Image of the shelf
            text_prompts: Optional text prompts for Grounding DINO / YOLO-World.
                          Defaults to reference product names.
            return_masks: If True, run SAM2 segmentation on detections.

        Returns:
            ShelfMatchResult with all matches
        """
        if isinstance(shelf_image, (str, Path)):
            shelf_path = str(shelf_image)
            shelf_img = Image.open(shelf_path).convert("RGB")
        else:
            shelf_path = "memory"
            shelf_img = shelf_image

        if not self._reference_products:
            raise RuntimeError("No reference products loaded. Call load_references() first.")

        prompts = text_prompts or self._reference_names

        # Step 1: Detection
        logger.info("Step 1: Running detector (%s)", self.config.detector_type)
        detections = self.detector.detect(shelf_img, prompts)

        if not detections.detections:
            logger.info("No detections found")
            return ShelfMatchResult(
                shelf_image_path=shelf_path,
                matches=[],
                num_references=len(self._reference_products),
            )

        # Step 2: Segmentation
        detected_bboxes = [d.bbox for d in detections.detections]
        detected_masks: list[np.ndarray] = []

        if return_masks and self.config.run_segmentation:
            logger.info("Step 2: Running SAM2 segmentation on %d detections", len(detected_bboxes))
            seg_output = self.segmenter.segment_automatic(shelf_img)
            # Associate each detection with nearest mask by IoU
            detected_masks = self._associate_masks(
                shelf_img.size, detected_bboxes, seg_output.masks
            )
        else:
            detected_masks = [None] * len(detected_bboxes)

        # Step 3: Feature matching
        logger.info("Step 3: Matching detected regions to references")
        matches = self.matcher.match(
            shelf_img, detected_bboxes, detected_masks
        )

        return ShelfMatchResult(
            shelf_image_path=shelf_path,
            matches=matches,
            num_references=len(self._reference_products),
        )

    def _associate_masks(
        self,
        image_size: tuple[int, int],
        bboxes: list[np.ndarray],
        masks: list[MaskResult],
    ) -> list[np.ndarray]:
        """Associate each detection bbox with the best matching SAM mask by IoU."""
        w, h = image_size
        result_masks: list[Optional[np.ndarray]] = [None] * len(bboxes)

        for i, bbox in enumerate(bboxes):
            best_iou = 0.0
            best_mask: Optional[np.ndarray] = None

            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)

            for m in masks:
                if m.mask is None:
                    continue
                # Check if mask intersects with bbox
                mask_h, mask_w = m.mask.shape
                if mask_h == 0 or mask_w == 0:
                    continue

                # Scale mask to image size
                # For simplicity, check overlap via downscaled mask
                try:
                    import cv2
                    mask_resized = cv2.resize(
                        m.mask.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)

                    bbox_area = (x2 - x1) * (y2 - y1)
                    intersect = mask_resized[y1:y2, x1:x2].sum()
                    union = bbox_area + mask_resized[y1:y2, x1:x2].sum() - intersect
                    iou = intersect / union if union > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_mask = m.mask
                except Exception:
                    continue

            result_masks[i] = best_mask

        return result_masks

    # ─── Persistence ──────────────────────────────────────────────

    def save_index(self, path: str | Path):
        """Save FAISS index and reference metadata to disk."""
        import pickle
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.matcher._index is not None:
            faiss.write_index(self.matcher._index, str(path.with_suffix(".index")))

        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self._reference_products, f)

        logger.info("Saved index to %s", path)

    def load_index(self, path: str | Path):
        """Load FAISS index and reference metadata from disk."""
        import pickle
        import faiss

        path = Path(path)

        if path.with_suffix(".index").exists():
            self.matcher._index = faiss.read_index(str(path.with_suffix(".index")))

        if path.with_suffix(".pkl").exists():
            with open(path.with_suffix(".pkl"), "rb") as f:
                self._reference_products = pickle.load(f)
            self._reference_names = [p.name for p in self._reference_products]
            self.matcher._reference_products = self._reference_products

        logger.info("Loaded index from %s", path)
