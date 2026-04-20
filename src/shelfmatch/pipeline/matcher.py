"""
Product matching module using SigLIP / DINOv2 + FAISS.

Workflow:
1. Extract features from reference product images
2. Build FAISS index
3. For each detected/cropped region, extract features
4. Find nearest reference via FAISS cosine similarity
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
class ReferenceProduct:
    """A reference product with its visual features."""
    product_id: str
    name: str
    features: np.ndarray  # L2-normalized feature vector
    reference_image: Image.Image
    multi_angle_features: list[np.ndarray] = field(default_factory=list)

    @property
    def feature(self) -> np.ndarray:
        """Average multi-angle features if available, else single feature."""
        if self.multi_angle_features:
            # Mean-pool multiple angles for robustness
            stacked = np.stack(self.multi_angle_features + [self.features])
            return stacked / np.linalg.norm(stacked, axis=1, keepdims=True)
        return self.features


@dataclass
class MatchResult:
    """A single match between a detected region and a reference product."""
    product_id: str
    product_name: str
    similarity: float  # cosine similarity
    detected_bbox: np.ndarray
    segmentation_mask: Optional[np.ndarray] = None
    confidence: str  # "high", "medium", "low"


class FeatureExtractor:
    """
    Extract visual features using SigLIP or DINOv2.

    SigLIP — better for product matching (trained with stronger contrastive loss)
    DINOv2 — better for fine-grained visual similarity
    """

    def __init__(
        self,
        model_type: str = "siglip",  # "siglip" | "dinov2"
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

        if model_type == "siglip":
            self.model_id = model_id or "google/siglip-base-patch16-224"
        elif model_type == "dinov2":
            self.model_id = model_id or "facebook/dinov2-base"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _load_model(self):
        if self._model is not None:
            return

        if self.model_type == "siglip":
            from transformers import AutoModel, AutoProcessor
            logger.info("Loading SigLIP: %s", self.model_id)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)
        elif self.model_type == "dinov2":
            from transformers import AutoImageProcessor, AutoModel
            logger.info("Loading DINOv2: %s", self.model_id)
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id)

        self._model.to(self.device)
        self._model.eval()

    def _extract_single(self, image: Image.Image) -> np.ndarray:
        """Extract feature vector from a single image."""
        self._load_model()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.model_type == "siglip":
                # SigLIP outputs logits per image-text pair; we want image features
                image_embeds = self._model.get_image_features(**inputs)
            else:  # dinov2
                image_embeds = self._model(**inputs).last_hidden_state
                # Average pool all patches
                image_embeds = image_embeds.mean(dim=1)

        # L2 normalize
        feat = image_embeds.cpu().numpy()
        feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
        return feat[0]

    def extract(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Extract features from multiple images."""
        return [self._extract_single(img) for img in images]

    def extract_crop(self, image: Image.Image, bbox: np.ndarray) -> np.ndarray:
        """
        Extract features from a cropped region.

        Args:
            image: Full PIL Image
            bbox: [x1, y1, x2, y2] normalized 0-1
        """
        w, h = image.size
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
        crop = image.crop((x1, y1, x2, y2))
        return self._extract_single(crop)


class ProductMatcher:
    """
    Match detected regions to reference products using FAISS + feature similarity.

    Pipeline:
    1. build_index(reference_products)  — one-time
    2. match(detected_regions)           — per shelf image
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        similarity_threshold_high: float = 0.85,
        similarity_threshold_medium: float = 0.70,
    ):
        self.extractor = feature_extractor or FeatureExtractor()
        self.threshold_high = similarity_threshold_high
        self.threshold_medium = similarity_threshold_medium
        self._index: Optional[faiss.Index] = None
        self._reference_products: list[ReferenceProduct] = []
        self._product_ids: list[str] = []

    def build_index(self, products: list[ReferenceProduct]):
        """
        Build FAISS index from reference products.

        Args:
            products: List of ReferenceProduct objects
        """
        import faiss

        self._reference_products = products

        # Stack all multi-angle features
        all_ids: list[str] = []
        all_feats: list[np.ndarray] = []

        for p in products:
            feat = p.feature
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            all_feats.append(feat)
            all_ids.append(p.product_id)

        # Flatten (take mean if multi-angle already pooled)
        features = np.vstack(all_feats).astype("float32")

        dim = features.shape[1]
        logger.info("Building FAISS index with %d products, dim=%d", len(products), dim)

        # Inner product (cosine similarity since features are L2-normalized)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(features)

        # Store mapping of FAISS index position → product
        self._product_ids = all_ids
        logger.info("FAISS index built with %d entries", self._index.ntotal)

    @torch.no_grad()
    def match(
        self,
        shelf_image: Image.Image,
        detected_bboxes: list[np.ndarray],
        detected_masks: Optional[list[np.ndarray]] = None,
        k: int = 1,
    ) -> list[MatchResult]:
        """
        Match detected regions to reference products.

        Args:
            shelf_image: Full shelf PIL Image
            detected_bboxes: List of [x1,y1,x2,y2] normalized bboxes
            detected_masks: Optional list of segmentation masks (HxW bool)
            k: Number of nearest neighbors to consider

        Returns:
            List of MatchResult, one per detected bbox
        """
        if self._index is None:
            raise RuntimeError("Must call build_index() first")

        import faiss

        # Extract features for each detected region
        region_feats = []
        for bbox in detected_bboxes:
            feat = self.extractor.extract_crop(shelf_image, bbox)
            region_feats.append(feat.astype("float32"))

        if not region_feats:
            return []

        region_feats = np.vstack(region_feats)

        # FAISS k-NN search
        D, I = self._index.search(region_feats, k)

        results = []
        for i, (dists, indices) in enumerate(zip(D, I)):
            top_idx = indices[0]
            top_dist = dists[0]

            # Cosine similarity from inner product
            similarity = float(top_dist)

            if similarity >= self.threshold_high:
                confidence = "high"
            elif similarity >= self.threshold_medium:
                confidence = "medium"
            else:
                confidence = "low"

            product = self._reference_products[top_idx]
            results.append(
                MatchResult(
                    product_id=product.product_id,
                    product_name=product.name,
                    similarity=similarity,
                    detected_bbox=detected_bboxes[i],
                    segmentation_mask=detected_masks[i] if detected_masks else None,
                    confidence=confidence,
                )
            )

        logger.info(
            "Matched %d detections: %d high, %d medium, %d low confidence",
            len(results),
            sum(1 for r in results if r.confidence == "high"),
            sum(1 for r in results if r.confidence == "medium"),
            sum(1 for r in results if r.confidence == "low"),
        )
        return results

    def add_references(self, new_products: list[ReferenceProduct]):
        """Add new reference products to the existing index."""
        import faiss

        all_feats = []
        all_ids = []
        for p in new_products:
            feat = p.feature
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            all_feats.append(feat)
            all_ids.append(p.product_id)

        features = np.vstack(all_feats).astype("float32")
        self._index.add(features)
        self._reference_products.extend(new_products)
        self._product_ids.extend(all_ids)

        logger.info("Added %d new products, total: %d", len(new_products), self._index.ntotal)
