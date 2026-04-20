"""
AutoTrainer — autonomous fine-tuning trigger for ShelfMatch.

Watches results.tsv. When F1 plateaus (N consecutive "discard"),
fires a self-training cycle automatically. Pushes new weights to git.

Design principles:
- Only trains when actually stuck (not on every run)
- Keeps all training artifacts versioned in git
- Fully self-contained: cron triggers it, it decides whether to train
- Idempotent: safe to run multiple times

Usage (cron):
  0 6 * * * cd ~/Projects/shelfmatch && \
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    HF_TOKEN=... \
    .venv/bin/python -m shelfmatch.training.auto_trainer

The daily pipeline test should also write to results.tsv so AutoTrainer
can react to it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Threading fix — must be before torch imports
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shelfmatch.training.self_train import run_self_training_cycle

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

RESULTS_TSV = "results.tsv"
CONFIG_YAML = "config.yaml"
WORKSPACE = "workspace"
MODELS_DIR = f"{WORKSPACE}/models"
SYNTHETIC_DIR = f"{WORKSPACE}/data/synthetic"
CONTRASTIVE_PAIRS = f"{SYNTHETIC_DIR}/contrastive_pairs.jsonl"
CROPS_JSONL = f"{SYNTHETIC_DIR}/crops.jsonl"


class AutoTrainer:
    """
    Autonomous fine-tuning trigger.

    Monitors results.tsv for F1 plateau. When detected, runs a full
    self-training cycle and commits the new weights.
    """

    def __init__(
        self,
        workspace: str = WORKSPACE,
        plateau_threshold: int = 5,
        min_synthetic_for_training: int = 10,
        model_name: str = "google/siglip-base-patch16-224",
        epochs: int = 5,
        lr: float = 1e-4,
        rank: int = 16,
        batch_size: int = 4,
        num_synthetic: int = 30,
        min_products: int = 2,
        max_products: int = 6,
        output_iter_prefix: str = "iter",
        dry_run: bool = False,
    ):
        self.workspace = Path(workspace)
        self.plateau_threshold = plateau_threshold
        self.min_synthetic_for_training = min_synthetic_for_training
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.rank = rank
        self.batch_size = batch_size
        self.num_synthetic = num_synthetic
        self.min_products = min_products
        self.max_products = max_products
        self.output_iter_prefix = output_iter_prefix
        self.dry_run = dry_run

    # ─── Results analysis ─────────────────────────────────────────────────────

    def read_results(self) -> list[dict]:
        """Read results.tsv and return list of experiment dicts."""
        tsv_path = self.workspace / RESULTS_TSV
        if not tsv_path.exists():
            return []

        lines = tsv_path.read_text().strip().split("\n")
        if len(lines) <= 1:
            return []

        header = lines[0].strip().split("\t")
        results = []
        for line in lines[1:]:
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                continue
            row = dict(zip(header, fields))
            # Cast numeric fields
            for col in ("precision", "recall", "f1", "runtime_seconds",
                        "num_detections", "num_matches"):
                if col in row and row[col]:
                    try:
                        row[col] = float(row[col])
                    except ValueError:
                        pass
            results.append(row)
        return results

    def detect_plateau(self, results: list[dict]) -> bool:
        """
        Return True if the last N experiments are all 'discard'
        and we have enough experiments to judge a plateau.
        """
        if len(results) < self.plateau_threshold:
            return False

        recent = results[-self.plateau_threshold:]
        all_discarded = all(r.get("status") == "discard" for r in recent)
        return all_discarded

    def get_last_f1(self, results: list[dict]) -> float | None:
        if not results:
            return None
        try:
            return float(results[-1].get("f1", 0))
        except (ValueError, TypeError):
            return None

    def get_best_f1(self, results: list[dict]) -> float:
        """Get the best F1 score seen so far."""
        best = 0.0
        for r in results:
            try:
                f1 = float(r.get("f1", 0))
                if f1 > best:
                    best = f1
            except (ValueError, TypeError):
                pass
        return best

    def get_latest_model_iter(self) -> int:
        """Find the highest iteration number in models dir."""
        models_dir = self.workspace / "models"
        if not models_dir.exists():
            return 0
        max_iter = 0
        for d in models_dir.iterdir():
            if d.is_dir() and d.name.startswith(self.output_iter_prefix):
                try:
                    num = int(d.name.split("-")[1])
                    if num > max_iter:
                        max_iter = num
                except (ValueError, IndexError):
                    pass
        return max_iter

    # ─── Synthetic data ──────────────────────────────────────────────────────

    def ensure_synthetic_data(self) -> bool:
        """
        Ensure we have enough synthetic contrastive pairs.
        If not, generate more.

        Returns True if we have enough data for training.
        """
        # Check if pairs file exists and has enough pairs
        pairs_path = self.workspace / CONTRASTIVE_PAIRS
        if pairs_path.exists():
            with open(pairs_path) as f:
                count = sum(1 for _ in f)
            logger.info("Found %d existing contrastive pairs", count)
            if count >= self.min_synthetic_for_training:
                return True
            logger.info("Not enough pairs (%d < %d), regenerating...",
                        count, self.min_synthetic_for_training)

        # Generate new synthetic data
        logger.info("Generating %d synthetic shelf images...", self.num_synthetic)
        return self._run_data_generation()

    def _run_data_generation(self) -> bool:
        """Generate synthetic data using the pipeline."""
        try:
            from shelfmatch.training.synthetic import SyntheticShelfGenerator
            from shelfmatch.training.formatter import TrainingFormatter

            test_dir = self.workspace / "data" / "test"
            ref_images = sorted(test_dir.glob("product_*.jpg"))
            if not ref_images:
                logger.error("No reference product images in %s", test_dir)
                return False

            ref_paths = [str(p) for p in ref_images]
            product_names = [p.stem for p in ref_images]

            # ── Generate synthetic shelves ──────────────────────────────────
            synthetic_out = self.workspace / SYNTHETIC_DIR
            synthetic_out.mkdir(parents=True, exist_ok=True)

            gen = SyntheticShelfGenerator(output_dir=str(synthetic_out))
            for name, path in zip(product_names, ref_paths):
                gen.add_product(name, path)
            gen.add_solid_background(color=(180, 170, 160), size=(800, 600))

            images_dir = synthetic_out / "images"
            annotations_dir = synthetic_out / "annotations"
            images_dir.mkdir(parents=True, exist_ok=True)
            annotations_dir.mkdir(parents=True, exist_ok=True)

            gen.generate_dataset(
                output_dir=str(synthetic_out),
                num_images=self.num_synthetic,
                min_products=self.min_products,
                max_products=self.max_products,
            )
            logger.info("Generated %d synthetic images", self.num_synthetic)

            # ── Build contrastive pairs ───────────────────────────────────
            formatter = TrainingFormatter(crop_margin=0.05)

            crops_count = formatter.process_synthetic_dataset(
                images_dir=str(images_dir),
                annotations_dir=str(annotations_dir),
                output_path=str(synthetic_out / "crops.jsonl"),
                mode="product_crop",
            )
            logger.info("Extracted %d product crops", crops_count)

            pairs_count = formatter.generate_contrastive_pairs(
                jsonl_path=str(synthetic_out / "crops.jsonl"),
                output_path=str(synthetic_out / "contrastive_pairs.jsonl"),
                num_negatives_per_positive=4,
            )
            logger.info("Built %d contrastive pairs", pairs_count)

            return pairs_count >= self.min_synthetic_for_training

        except Exception as e:
            logger.error("Synthetic data generation failed: %s", e, exc_info=True)
            return False

    # ─── Fine-tuning ──────────────────────────────────────────────────────────

    def run_fine_tuning(self) -> bool:
        """
        Run one full self-training cycle.

        Returns True on success, False on failure.
        """
        latest_iter = self.get_latest_model_iter()
        new_iter = latest_iter + 1
        output_dir = self.workspace / "models" / f"{self.output_iter_prefix}-{new_iter:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        test_dir = self.workspace / "data" / "test"
        ref_images = sorted(test_dir.glob("product_*.jpg"))
        test_images = sorted(test_dir.glob("shelf_*.jpg"))

        if not ref_images:
            logger.error("No reference images found")
            return False

        ref_paths = [str(p) for p in ref_images]
        test_paths = [str(p) for p in test_images] if test_images else []

        logger.info("=== Starting self-training iteration %d ===", new_iter)
        t0 = time.time()

        try:
            summary = run_self_training_cycle(
                reference_images=ref_paths,
                test_shelf_images=test_paths,
                output_dir=str(output_dir),
                num_synthetic=self.num_synthetic,
                model_name=self.model_name,
                epochs=self.epochs,
                lr=self.lr,
                rank=self.rank,
                batch_size=self.batch_size,
            )
            elapsed = time.time() - t0
            logger.info(
                "Self-training complete in %.1fs — pairs=%d, model=%s",
                elapsed, summary.get("contrastive_pairs", 0),
                summary.get("model_path", "unknown"),
            )

            # Save training summary
            with open(output_dir / "training_summary.json", "w") as f:
                json.dump({**summary, "iteration": new_iter}, f, indent=2)

            return True

        except Exception as e:
            logger.error("Self-training failed: %s", e, exc_info=True)
            return False

    # ─── Git ─────────────────────────────────────────────────────────────────

    def git_commit_model(self, output_dir: Path, f1_before: float, f1_after: float | None):
        """Commit new model weights and training summary to git."""
        try:
            # Stage everything under workspace/models
            subprocess.run(["git", "add", str(output_dir)], check=True)
            result = subprocess.run(
                ["git", "status", "--porcelain", str(output_dir)],
                capture_output=True, text=True,
            )
            if not result.stdout.strip():
                logger.info("No changes to commit in %s", output_dir)
                return

            msg = (
                f"Train iter {output_dir.name}: F1 {f1_before:.4f}"
                + (f" → {f1_after:.4f}" if f1_after else " (new training)")
            )
            subprocess.run(["git", "commit", "-m", msg], check=True)
            logger.info("Committed: %s", msg)
        except subprocess.CalledProcessError as e:
            logger.warning("Git commit failed: %s", e.stderr)

    # ─── Main decision loop ───────────────────────────────────────────────────

    def should_train(self, results: list[dict]) -> bool:
        """Decide whether to run training right now."""
        if not results:
            logger.info("No results yet — skipping training")
            return False

        if not self.detect_plateau(results):
            last_f1 = self.get_last_f1(results)
            best_f1 = self.get_best_f1(results)
            logger.info(
                "F1=%.4f (best=%.4f) — not plateaued yet, skipping training",
                last_f1, best_f1,
            )
            return False

        logger.info("⚠️  F1 plateau detected (%d consecutive discards) — will train",
                     self.plateau_threshold)
        return True

    def run_once(self) -> dict:
        """
        Run one complete AutoTrainer cycle.

        Returns a summary dict.
        """
        results = self.read_results()
        f1_before = self.get_last_f1(results) or 0.0
        best_before = self.get_best_f1(results)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results),
            "f1_before": f1_before,
            "best_f1_before": best_before,
            "plateau_detected": False,
            "training_triggered": False,
            "training_success": False,
            "f1_after": None,
        }

        # Decision: should we train?
        if not self.should_train(results):
            if self.dry_run:
                logger.info("[DRY RUN] Would NOT train right now")
            return summary

        if self.dry_run:
            logger.info("[DRY RUN] Would train now")
            summary["training_triggered"] = True
            return summary

        # Ensure we have synthetic data
        summary["plateau_detected"] = True
        summary["training_triggered"] = True

        if not self.ensure_synthetic_data():
            logger.error("Could not generate sufficient synthetic data — aborting training")
            return summary

        # Run training
        success = self.run_fine_tuning()
        summary["training_success"] = success

        if success:
            # Commit new weights
            latest_iter = self.get_latest_model_iter()
            output_dir = self.workspace / "models" / f"{self.output_iter_prefix}-{latest_iter:03d}"
            self.git_commit_model(output_dir, f1_before, None)
            summary["f1_after"] = None  # Would need re-eval to know
            summary["model_dir"] = str(output_dir)

        return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="ShelfMatch AutoTrainer")
    parser.add_argument("--workspace", default=".", help="Project workspace root")
    parser.add_argument("--plateau-threshold", type=int, default=5,
                        help="Consecutive discards before triggering training")
    parser.add_argument("--min-synthetic", type=int, default=10,
                        help="Minimum contrastive pairs before training")
    parser.add_argument("--num-synthetic", type=int, default=30,
                        help="Synthetic shelf images per training cycle")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print what would happen, don't train")
    args = parser.parse_args()

    trainer = AutoTrainer(
        workspace=args.workspace,
        plateau_threshold=args.plateau_threshold,
        min_synthetic_for_training=args.min_synthetic,
        num_synthetic=args.num_synthetic,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    logger.info("AutoTrainer starting — workspace=%s", args.workspace)
    summary = trainer.run_once()
    logger.info("Result: %s", json.dumps(summary, indent=2))

    if summary["training_triggered"] and summary["training_success"]:
        print("\n✅ Fine-tuning completed successfully")
        print(f"   Model: {summary.get('model_dir', 'unknown')}")
    elif summary["plateau_detected"] and not summary["training_triggered"]:
        print("\n⏭️  Plateau detected but dry-run mode")


if __name__ == "__main__":
    main()
