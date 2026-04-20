"""
Gradio web app for ShelfMatch.

Usage:
    cd ~/Projects/shelfmatch
    uv run python -m shelfmatch.webapp.main
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import logging

# Add src to path so we can import shelfmatch
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shelfmatch.pipeline.shelfmatcher import ShelfMatcher, PipelineConfig
from shelfmatch.pipeline.matcher import ReferenceProduct
from shelfmatch.pipeline.matcher import FeatureExtractor

logger = logging.getLogger(__name__)

# ─── Visualization ──────────────────────────────────────────────────────────────

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 255), (255, 128, 0), (0, 200, 200),
]


def draw_results(
    image: Image.Image,
    matches: list,
    show_mask: bool = True,
    alpha: float = 0.35,
) -> Image.Image:
    """
    Draw bounding boxes, labels, and optionally masks on the shelf image.

    Args:
        image: PIL Image of the shelf
        matches: List of MatchResult objects
        show_mask: Whether to overlay segmentation masks
        alpha: Mask overlay transparency

    Returns:
        Annotated PIL Image
    """
    img = np.array(image).copy()
    w, h = image.size

    for i, match in enumerate(matches):
        color = COLORS[i % len(COLORS)]
        x1 = int(match.detected_bbox[0] * w)
        y1 = int(match.detected_bbox[1] * h)
        x2 = int(match.detected_bbox[2] * w)
        y2 = int(match.detected_bbox[3] * h)

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw mask overlay
        if show_mask and match.segmentation_mask is not None:
            mask = match.segmentation_mask.astype(np.uint8)
            # Resize mask to image dimensions
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            colored_mask = np.zeros_like(img)
            colored_mask[:, :] = color
            img = np.where(
                mask_resized[:, :, None].astype(bool),
                (img * (1 - alpha) + colored_mask * alpha).astype(np.uint8),
                img,
            )

        # Label background
        label = f"{match.product_name} ({match.confidence})"
        label += f" | sim: {match.similarity:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 8, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    return Image.fromarray(img)


def results_to_text(matches: list) -> str:
    """Format match results as readable text."""
    if not matches:
        return "No products detected."

    lines = [f"## Detection Results ({len(matches)} products found)\n"]
    for i, m in enumerate(matches, 1):
        lines.append(
            f"**{i}. {m.product_name}** — {m.confidence.upper()} confidence  "
            f"(similarity: {m.similarity:.3f})"
        )
        lines.append(
            f"   BBox: [{m.detected_bbox[0]:.3f}, {m.detected_bbox[1]:.3f}, "
            f"{m.detected_bbox[2]:.3f}, {m.detected_bbox[3]:.3f}]"
        )
        if m.segmentation_mask is not None:
            lines.append(f"   Mask: {m.segmentation_mask.sum():,} pixels")
        lines.append("")

    return "\n".join(lines)


# ─── Gradio Interface ──────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Build the full Gradio interface."""

    with gr.Blocks(
        title="ShelfMatch",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            "# ShelfMatch 🛒\n"
            "Reference-based retail shelf product detection"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Reference Products")
                ref_upload = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Upload reference product images",
                )
                ref_names = gr.Textbox(
                    label="Product names (one per line, same order)",
                    placeholder="Product A\nProduct B\nProduct C",
                    lines=3,
                )
                load_btn = gr.Button("Load References", variant="primary")

                gr.Markdown("### 2. Configuration")
                detector_dropdown = gr.Dropdown(
                    ["grounding_dino", "owlv2", "yoloworld"],
                    value="grounding_dino",
                    label="Detector",
                )
                feature_dropdown = gr.Dropdown(
                    ["siglip", "dinov2"],
                    value="siglip",
                    label="Feature Model",
                )
                box_thresh = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.35, step=0.05,
                    label="Detection Threshold",
                )
                sim_thresh_high = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.85, step=0.05,
                    label="High-Similarity Threshold",
                )
                show_mask = gr.Checkbox(label="Show Segmentation Masks", value=True)

                gr.Markdown("### 3. Shelf Image")
                shelf_upload = gr.Image(
                    sources=["upload", "clipboard"],
                    type="pil",
                    label="Upload shelf image",
                )
                detect_btn = gr.Button("Detect Products", variant="primary")

                status_msg = gr.Textbox(label="Status", lines=2)

            with gr.Column(scale=2):
                result_gallery = gr.Image(
                    label="Detection Results",
                    type="pil",
                )
                result_text = gr.Markdown("")

        # State
        matcher_state = gr.State()

        # ── Logic ────────────────────────────────────────────────────

        def load_references(ref_files, ref_names_text, detector, feature, box_thresh, sim_thresh):
            """Load reference products and build matcher index."""
            try:
                if not ref_files:
                    return "⚠️ Please upload at least one reference image.", None

                paths = [f.name for f in ref_files]
                names = (
                    [n.strip() for n in ref_names_text.split("\n") if n.strip()]
                    if ref_names_text
                    else None
                )

                config = PipelineConfig(
                    detector_type=detector,
                    feature_model=feature,
                    box_threshold=box_thresh,
                    similarity_threshold_high=sim_thresh,
                )
                matcher = ShelfMatcher(config=config)

                # Group single images per product
                ref_groups = [[p] for p in paths]
                matcher.load_references(ref_groups, names)

                return (
                    f"✅ Loaded {len(paths)} reference products. Index ready.",
                    matcher,
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"❌ Error loading references: {e}", None

        def run_detection(matcher, shelf_img, show_mask_val, box_thresh, sim_thresh):
            """Run detection pipeline on the shelf image."""
            try:
                if matcher is None:
                    return None, "⚠️ Load reference products first."

                # Update thresholds
                matcher.config.box_threshold = box_thresh
                matcher.config.similarity_threshold_high = sim_thresh

                results = matcher.detect(shelf_img, return_masks=show_mask_val)

                if not results.matches:
                    return shelf_img, "No products detected on this shelf."

                annotated = draw_results(shelf_img, results.matches, show_mask=show_mask_val)
                text = results_to_text(results.matches)
                return annotated, text

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Detection error: {e}"

        # Wire up
        load_btn.click(
            load_references,
            inputs=[ref_upload, ref_names, detector_dropdown, feature_dropdown, box_thresh, sim_thresh_high],
            outputs=[status_msg, matcher_state],
        )

        detect_btn.click(
            run_detection,
            inputs=[matcher_state, shelf_upload, show_mask, box_thresh, sim_thresh_high],
            outputs=[result_gallery, result_text],
        )

    return app


def main():
    logging.basicConfig(level=logging.INFO)
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
