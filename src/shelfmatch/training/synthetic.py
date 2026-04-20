"""
Synthetic shelf image generator for zero-label training data.

Generates training data by:
1. Taking reference product images
2. Compositing them onto simulated shelf backgrounds
3. Randomizing lighting, rotation, scale, occlusion
4. Auto-generating bounding box annotations

This is the key to near-zero labeling: instead of annotating shelf images,
you use your known reference products to generate synthetic training data.
"""
from __future__ import annotations

import random
import logging
from pathlib import Path
from typing import Optional
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

logger = logging.getLogger(__name__)


class SyntheticShelfGenerator:
    """
    Generate synthetic shelf images for training product detection models.

    Usage:
        generator = SyntheticShelfGenerator()
        generator.add_product("product_a", "path/to/product_a.jpg")
        generator.add_product("product_b", "path/to/product_b.jpg")
        image, annotations = generator.generate(num_products=4, background="shelf_bg.jpg")
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.rng = random.Random(random_seed)

        self._products: dict[str, Image.Image] = {}
        self._shelf_backgrounds: list[Image.Image] = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_product(self, product_id: str, image_path: str):
        """Add a reference product image."""
        img = Image.open(image_path).convert("RGB")
        self._products[product_id] = img
        logger.info("Added product '%s' (%dx%d)", product_id, *img.size)

    def add_background(self, image_path: str):
        """Add a shelf background image (empty shelf or generic shelf texture)."""
        img = Image.open(image_path).convert("RGB")
        self._shelf_backgrounds.append(img)
        logger.info("Added background (%dx%d)", *img.size)

    def add_solid_background(self, color: tuple[int, int, int], size: tuple[int, int]):
        """Create a solid color background as fallback."""
        img = Image.new("RGB", size, color)
        self._shelf_backgrounds.append(img)

    def _apply_augmentation(self, product_img: Image.Image) -> Image.Image:
        """Apply random augmentations to simulate real-world variation."""
        img = product_img.copy()

        # Random brightness
        factor = self.rng.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

        # Random contrast
        factor = self.rng.uniform(0.7, 1.3)
        img = ImageEnhance.Contrast(img).enhance(factor)

        # Random color
        factor = self.rng.uniform(0.8, 1.2)
        img = ImageEnhance.Color(img).enhance(factor)

        # Random blur (10% chance)
        if self.rng.random() < 0.1:
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Random rotation (±15°)
        angle = self.rng.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))

        # Random scale (0.8x - 1.2x)
        scale = self.rng.uniform(0.8, 1.2)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        # Don't scale below 32px
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        return img

    def _find_placement(
        self,
        shelf_img: Image.Image,
        product_w: int,
        product_h: int,
        existing_bboxes: list[list[int]],
        margin: int = 10,
    ) -> Optional[tuple[int, int]]:
        """
        Find a non-overlapping placement on the shelf.
        Uses a simple grid-based approach.
        """
        w, h = shelf_img.size

        # Try random positions first
        for _ in range(50):
            x = self.rng.randint(margin, w - product_w - margin)
            y = self.rng.randint(margin, h - product_h - margin)

            new_box = [x, y, x + product_w, y + product_h]
            overlaps = False
            for eb in existing_bboxes:
                if self._iou(new_box, eb) > 0.1:
                    overlaps = True
                    break
            if not overlaps:
                return x, y

        return None  # No space found

    def _iou(self, box_a: list[int], box_b: list[int]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def generate(
        self,
        num_products: int = 5,
        background: Optional[Image.Image] = None,
        occlude: bool = True,
        output_prefix: str = "syn_shelf",
    ) -> tuple[Image.Image, dict]:
        """
        Generate a single synthetic shelf image.

        Args:
            num_products: How many products to place
            background: Background image. Random from pool if None
            occlude: Whether to allow partial occlusions
            output_prefix: Prefix for saved files

        Returns:
            (composited_image, annotations_dict)
        """
        # Pick background
        if background is None:
            if self._shelf_backgrounds:
                bg = self.rng.choice(self._shelf_backgrounds).copy()
            else:
                # Default gray shelf
                bg = Image.new("RGB", (800, 600), (180, 170, 160))
        else:
            bg = background.copy()

        bg_arr = np.array(bg)
        if len(bg_arr.shape) == 2:
            bg = Image.fromarray(np.stack([bg_arr] * 3, axis=-1))
        bg_w, bg_h = bg.size

        placed: list[list[int]] = []
        annotations: dict = {
            "image": f"{output_prefix}.jpg",
            "annotations": [],
        }

        available_products = list(self._products.items())

        for i in range(num_products):
            if not available_products:
                break

            product_id, product_template = self.rng.choice(available_products)

            # Apply augmentation
            aug_product = self._apply_augmentation(product_template)

            pw, ph = aug_product.size

            # Scale product relative to shelf
            target_h = self.rng.randint(bg_h // 8, bg_h // 3)
            scale = target_h / ph
            new_pw, new_ph = int(pw * scale), int(ph * scale)
            new_pw = max(new_pw, 48)
            new_ph = max(new_ph, 48)
            aug_product = aug_product.resize((new_pw, new_ph), Image.BICUBIC)

            # Find placement
            pos = self._find_placement(bg, new_pw, new_ph, placed)
            if pos is None:
                logger.debug("Could not place product %s — no space", product_id)
                continue

            x, y = pos

            # Create mask for blending
            mask = aug_product.split()[-1] if aug_product.mode == "RGBA" else None

            if mask is not None:
                # White background for non-transparent areas
                background_for_product = Image.new("RGBA", aug_product.size, (255, 255, 255, 255))
                aug_product_rgba = Image.composite(aug_product, background_for_product, mask)
                bg.paste(aug_product_rgba.convert("RGB"), (x, y), mask)
            else:
                bg.paste(aug_product, (x, y))

            placed.append([x, y, x + new_pw, y + new_ph])

            # Annotations
            annotations["annotations"].append({
                "product_id": product_id,
                "bbox": {
                    "x1": x / bg_w,
                    "y1": y / bg_h,
                    "x2": (x + new_pw) / bg_w,
                    "y2": (y + new_ph) / bg_h,
                },
                "bbox_pixel": [x, y, x + new_pw, y + new_ph],
            })

        # Add noise / texture to whole image
        bg = ImageEnhance.Sharpness(bg).enhance(self.rng.uniform(0.8, 1.2))
        if self.rng.random() < 0.3:
            bg = bg.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Save
        if self.output_dir:
            bg.save(self.output_dir / f"{output_prefix}.jpg", quality=92)
            with open(self.output_dir / f"{output_prefix}.json", "w") as f:
                json.dump(annotations, f, indent=2)

        return bg, annotations

    def generate_dataset(
        self,
        num_images: int,
        min_products: int = 2,
        max_products: int = 8,
        output_dir: str,
    ):
        """
        Generate a full synthetic dataset.

        Args:
            num_images: Number of shelf images to generate
            min_products: Min products per shelf
            max_products: Max products per shelf
            output_dir: Where to save images + annotations
        """
        out = Path(output_dir)
        images_dir = out / "images"
        annotations_dir = out / "annotations"
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating %d synthetic shelf images → %s", num_images, out)

        for i in range(num_images):
            num_prods = self.rng.randint(min_products, max_products)
            img, ann = self.generate(
                num_products=num_prods,
                output_prefix=f"syn_{i:04d}",
            )

            # Save to output dir
            img.save(images_dir / f"syn_{i:04d}.jpg", quality=92)
            with open(annotations_dir / f"syn_{i:04d}.json", "w") as f:
                json.dump(ann, f, indent=2)

            if (i + 1) % 100 == 0:
                logger.info("  Generated %d / %d images", i + 1, num_images)

        # Generate dataset index
        index = {
            "num_images": num_images,
            "products": list(self._products.keys()),
            "split": "synthetic",
        }
        with open(out / "dataset.json", "w") as f:
            json.dump(index, f, indent=2)

        logger.info("Dataset generation complete: %d images", num_images)
        return out
