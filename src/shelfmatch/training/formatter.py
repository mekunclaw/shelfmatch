"""
Formatter: Synthetic shelf data → fine-tuning format for SigLIP.

The self-training loop generates synthetic shelf images with known product placements.
This formatter converts them into training examples for contrastive fine-tuning
of the SigLIP vision encoder.

Two modes:
  1. Product-crop mode: extract cropped product regions from synthetic shelves,
     pair with product_id label → contrastive pairs for SiFP-like training.
  2. Full-shelf mode: shelf image + list of (bbox, product_id) → 
     image-text contrastive pairs.

Output: JSONL with {image_path, product_id, bbox, label}
where label is the product_id string for contrastive training.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TrainingFormatter:
    """
    Convert synthetic shelf data into contrastive training examples.

    Usage:
        formatter = TrainingFormatter()
        formatter.process_synthetic_dataset(
            images_dir="workspace/data/synthetic/images",
            annotations_dir="workspace/data/synthetic/annotations",
            output_path="workspace/data/synthetic/train.jsonl",
            mode="product_crop",
        )
    """

    def __init__(self, crop_margin: float = 0.05):
        """
        Args:
            crop_margin: Extra margin around bbox when cropping (as fraction of bbox size).
        """
        self.crop_margin = crop_margin

    def process_synthetic_dataset(
        self,
        images_dir: str | Path,
        annotations_dir: str | Path,
        output_path: str | Path,
        mode: str = "product_crop",
    ) -> int:
        """
        Process a full synthetic dataset.

        Args:
            images_dir: Path to synthetic shelf images
            annotations_dir: Path to annotation JSON files
            output_path: Where to write the formatted JSONL
            mode: "product_crop" (extract product crops) or "full_shelf" (keep full image)

        Returns:
            Number of examples written
        """
        images_dir = Path(images_dir)
        annotations_dir = Path(annotations_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w") as outf:
            for ann_file in sorted(annotations_dir.glob("*.json")):
                img_file = images_dir / ann_file.with_suffix(".jpg").name
                if not img_file.exists():
                    img_file = images_dir / ann_file.with_suffix(".png").name
                if not img_file.exists():
                    logger.warning("No image for %s", ann_file)
                    continue

                examples = self.process_single(
                    image_path=str(img_file),
                    annotation_path=str(ann_file),
                    mode=mode,
                )
                for ex in examples:
                    outf.write(json.dumps(ex) + "\n")
                    count += 1

        logger.info("Wrote %d training examples → %s", count, output_path)
        return count

    def process_single(
        self,
        image_path: str | Path,
        annotation_path: str | Path,
        mode: str = "product_crop",
    ) -> list[dict]:
        """
        Process a single synthetic shelf image.

        Args:
            image_path: Path to shelf image
            annotation_path: Path to annotation JSON
            mode: "product_crop" or "full_shelf"

        Returns:
            List of training examples
        """
        image_path = Path(image_path)
        ann = json.loads(Path(annotation_path).read_text())

        examples = []
        for detection in ann.get("annotations", []):
            product_id = detection["product_id"]
            bbox = detection["bbox_pixel"]  # [x1, y1, x2, y2]

            if mode == "product_crop":
                ex = self._extract_crop(str(image_path), bbox, product_id)
            else:
                ex = {
                    "image_path": str(image_path),
                    "product_id": product_id,
                    "bbox": bbox,
                    "label": product_id,
                }
            examples.append(ex)

        return examples

    def _extract_crop(
        self,
        image_path: str,
        bbox: list[int],
        product_id: str,
    ) -> dict:
        """Extract a cropped product region from a shelf image."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        x1, y1, x2, y2 = bbox
        # Add margin
        mx = int((x2 - x1) * self.crop_margin)
        my = int((y2 - y1) * self.crop_margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        crop = img.crop((x1, y1, x2, y2))

        # Save cropped product image
        crop_hash = hash((image_path, x1, y1, x2, y2)) % 10_000_000
        crop_dir = Path(image_path).parent / ".." / "crops"
        crop_dir = (crop_dir / product_id)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"crop_{crop_hash:07d}.jpg"
        crop.save(crop_path, quality=88)

        return {
            "image_path": str(crop_path),
            "product_id": product_id,
            "bbox": bbox,
            "label": product_id,
            "original_image": str(image_path),
        }

    def generate_contrastive_pairs(
        self,
        jsonl_path: str | Path,
        output_path: str | Path,
        num_negatives_per_positive: int = 4,
    ) -> int:
        """
        Convert product-crop examples into contrastive pairs for SigLIP training.

        Each positive pair: (crop_i, crop_j) where i == j (same product)
        Negative pairs: (crop_i, crop_j) where i != j (different products)

        Output format: JSONL with {anchor_path, positive_path, negative_paths, anchor_label}

        Args:
            jsonl_path: Input from process_single/process_synthetic_dataset
            output_path: Where to write contrastive pairs
            num_negatives_per_positive: Number of negatives per anchor-positive pair

        Returns:
            Number of pairs written
        """
        import random

        # Load all examples
        examples = []
        with open(jsonl_path) as f:
            for line in f:
                examples.append(json.loads(line))

        # Group by product_id
        by_product: dict[str, list[dict]] = {}
        for ex in examples:
            pid = ex["product_id"]
            by_product.setdefault(pid, []).append(ex)

        product_ids = list(by_product.keys())
        logger.info(
            "Generating contrastive pairs: %d products, %d total crops",
            len(product_ids), len(examples),
        )

        count = 0
        with open(output_path, "w") as outf:
            for anchor_ex in examples:
                anchor_pid = anchor_ex["product_id"]
                anchor_path = anchor_ex["image_path"]

                # Positive: same product, different image (if available)
                positives = [ex for ex in by_product[anchor_pid]
                             if ex["image_path"] != anchor_path]
                if not positives:
                    continue  # need at least one positive

                positive = random.choice(positives)
                positive_path = positive["image_path"]

                # Negatives: different products
                negatives = [ex for ex in examples if ex["product_id"] != anchor_pid]
                if len(negatives) < num_negatives_per_positive:
                    continue

                neg_samples = random.sample(negatives, num_negatives_per_positive)
                negative_paths = [n["image_path"] for n in neg_samples]

                pair = {
                    "anchor_path": anchor_path,
                    "positive_path": positive_path,
                    "negative_paths": negative_paths,
                    "anchor_label": anchor_pid,
                    "positive_label": anchor_pid,
                    "negative_labels": [n["product_id"] for n in neg_samples],
                }
                outf.write(json.dumps(pair) + "\n")
                count += 1

        logger.info("Wrote %d contrastive pairs → %s", count, output_path)
        return count
