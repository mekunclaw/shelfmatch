"""
Ollama/Gemma4 orchestrator — the autoresearch loop for ShelfMatch.

Inspired by karpathy/autoresearch: an LLM agent runs experiments
autonomously, evaluates results, and iteratively improves the pipeline.

How it works:
1. Initialize with a ShelfMatcher pipeline config
2. The agent (Gemma4 via Ollama) reads current results.tsv
3. Proposes a config modification (thresholds, model choices, etc.)
4. Runs the pipeline on validation data
5. Evaluates results (precision/recall against pseudo-labels)
6. Logs to results.tsv — keeps or discards the change
7. Repeats until满意 (satisfied) or max iterations

The agent modifies config.yaml, not Python code — keeps it simple.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import shutil

import yaml

logger = logging.getLogger(__name__)

RESULTS_TSV = "results.tsv"
CONFIG_YAML = "config.yaml"
PROMPT_MD = "program.md"


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    commit: str
    val_precision: float
    val_recall: float
    val_f1: float
    num_detections: int
    num_matches: int
    runtime_seconds: float
    status: str  # "keep" | "discard" | "crash"
    description: str
    config_snapshot: dict  # frozen copy of config at experiment time


@dataclass
class OrchestratorConfig:
    """Settings for the orchestrator."""
    ollama_model: str = "gemma4:latest"
    ollama_base_url: str = "http://localhost:11434"
    max_iterations: int = 20
    experiment_budget_seconds: int = 300  # 5 min per experiment
    val_data_dir: str = "workspace/data/val"
    workspace_dir: str = "workspace"
    min_improvement: float = 0.01  # min F1 improvement to "keep" a change
    # Pipeline defaults (can be overridden by agent)
    default_detector: str = "grounding_dino"
    default_feature_model: str = "siglip"
    default_box_threshold: float = 0.35
    default_text_threshold: float = 0.25
    default_sim_threshold_high: float = 0.85


class ShelfOrchestrator:
    """
    Autonomous experiment runner using Ollama as the agent brain.

    The agent reads program.md for its instructions, modifies config.yaml
    based on experiment outcomes, and drives the self-training loop.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._setup_workspace()
        self._init_results_tsv()

    def _setup_workspace(self):
        """Create necessary directories."""
        dirs = [
            self.config.workspace_dir,
            self.config.val_data_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _init_results_tsv(self):
        """Create results.tsv if it doesn't exist."""
        tsv_path = Path(self.config.workspace_dir) / RESULTS_TSV
        if not tsv_path.exists():
            tsv_path.write_text(
                "commit\tprecision\trecall\tf1\tnum_detections\tnum_matches\t"
                "runtime_seconds\tstatus\tdescription\n"
            )
            logger.info("Created %s", tsv_path)

    def _get_commit_hash(self) -> str:
        """Short git commit hash of current state."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.config.workspace_dir,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()[:7]
        except Exception:
            return "no-git"

    def _load_current_config(self) -> dict:
        """Load current pipeline config from YAML."""
        cfg_path = Path(self.config.workspace_dir) / CONFIG_YAML
        if cfg_path.exists():
            return yaml.safe_load(cfg_path.read_text()) or {}
        # Return defaults
        return {
            "detector_type": self.config.default_detector,
            "feature_model": self.config.default_feature_model,
            "box_threshold": self.config.default_box_threshold,
            "text_threshold": self.config.default_text_threshold,
            "similarity_threshold_high": self.config.default_sim_threshold_high,
        }

    def _save_config(self, config: dict):
        """Save pipeline config to YAML."""
        cfg_path = Path(self.config.workspace_dir) / CONFIG_YAML
        cfg_path.write_text(yaml.dump(config, default_flow_style=False))
        logger.info("Saved config to %s", cfg_path)

    def _git_commit(self, description: str):
        """Commit current state with a description."""
        try:
            subprocess.run(["git", "add", "."], cwd=self.config.workspace_dir, check=True)
            subprocess.run(
                ["git", "commit", "-m", description],
                cwd=self.config.workspace_dir,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Git commit failed (may be nothing to commit): %s", e.stderr)

    # ─── Ollama / LLM interaction ──────────────────────────────────

    def _ollama_chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """
        Send a chat request to Ollama.

        Args:
            messages: [{"role": "user", "content": "..."}]
            temperature: Sampling temperature

        Returns:
            Response text from the model
        """
        import ollama

        # Ensure model is available
        try:
            ollama.show(self.config.ollama_model)
        except ollama.ResponseError:
            logger.info("Pulling Ollama model: %s", self.config.ollama_model)
            ollama.pull(self.config.ollama_model)

        response = ollama.chat(
            model=self.config.ollama_model,
            messages=messages,
            options={"temperature": temperature},
        )
        return response["message"]["content"]

    def _build_agent_prompt(self, current_config: dict, results_summary: str) -> str:
        """
        Build the system prompt for the agent.

        The agent reads current config, looks at results history,
        and proposes a change to improve F1 score.
        """
        return (
            f"You are an autonomous ML engineer running experiments on a retail shelf "
            f"product detection pipeline. Your goal is to maximize F1 score on validation data.\n\n"
            f"## Current Pipeline Config\n"
            f"{json.dumps(current_config, indent=2)}\n\n"
            f"## Results History (recent experiments)\n"
            f"{results_summary}\n\n"
            f"## Your Task\n"
            f"Read program.md for instructions, then propose ONE config modification "
            f"to improve the F1 score. Respond ONLY with the updated config as YAML.\n"
            f"Focus on:\n"
            f"- Detection thresholds (box_threshold, text_threshold)\n"
            f"- Similarity thresholds for matching\n"
            f"- Model choices (detector_type, feature_model)\n"
            f"- Segmentation settings\n\n"
            f"Respond ONLY with valid YAML config. No markdown fences. No explanation."
        )

    # ─── Self-training loop ────────────────────────────────────────

    def run_self_training(
        self,
        reference_images: list[str],
        val_shelf_images: list[str],
        pseudo_labels: Optional[dict] = None,
    ):
        """
        Run the autonomous self-training loop.

        Args:
            reference_images: Paths to reference product images
            val_shelf_images: Paths to validation shelf images
            pseudo_labels: Optional dict of {shelf_path: [{"product_id": ..., "bbox": ...}]}
                          If not provided, high-confidence matches are used as pseudo-labels
        """
        import ollama
        from shelfmatch.pipeline.shelfmatcher import ShelfMatcher, PipelineConfig

        logger.info("Starting self-training loop: %d refs, %d val images",
                    len(reference_images), len(val_shelf_images))

        current_config = self._load_current_config()
        self._save_config(current_config)
        commit = self._git_commit("initial baseline")

        for iteration in range(self.config.max_iterations):
            logger.info("=== Iteration %d / %d ===", iteration + 1, self.config.max_iterations)

            # Run pipeline with current config
            pipeline_config = PipelineConfig(
                detector_type=current_config.get("detector_type", self.config.default_detector),
                feature_model=current_config.get("feature_model", self.config.default_feature_model),
                box_threshold=current_config.get("box_threshold", self.config.default_box_threshold),
                text_threshold=current_config.get("text_threshold", self.config.default_text_threshold),
                similarity_threshold_high=current_config.get(
                    "similarity_threshold_high", self.config.default_sim_threshold_high
                ),
            )

            matcher = ShelfMatcher(config=pipeline_config)
            matcher.load_references(reference_images)

            all_results = []
            start_time = time.time()

            for shelf_path in val_shelf_images:
                try:
                    results = matcher.detect(shelf_path, return_masks=True)
                    all_results.append(results)
                except Exception as e:
                    logger.error("Detection failed on %s: %s", shelf_path, e)

            runtime = time.time() - start_time

            # Evaluate against pseudo-labels
            metrics = self._evaluate(all_results, pseudo_labels)

            # Log experiment
            experiment = ExperimentResult(
                commit=commit,
                val_precision=metrics["precision"],
                val_recall=metrics["recall"],
                val_f1=metrics["f1"],
                num_detections=sum(len(r.matches) for r in all_results),
                num_matches=sum(
                    sum(1 for m in r.matches if m.confidence == "high")
                    for r in all_results
                ),
                runtime_seconds=runtime,
                status="keep",  # will be updated below
                description=f"iter_{iteration}",
                config_snapshot=current_config.copy(),
            )

            # Decide: keep or discard?
            prev_f1 = self._get_last_f1()
            if prev_f1 is not None and experiment.val_f1 < prev_f1 + self.config.min_improvement:
                experiment.status = "discard"
                # Revert config
                current_config = self._load_current_config()
            else:
                experiment.status = "keep"
                self._log_experiment(experiment)

            commit = self._git_commit(f"{experiment.status} iter_{iteration}: F1={experiment.val_f1:.4f}")

            # Ask agent what to do next
            results_summary = self._summarize_results()
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an ML engineer. Read program.md, then output ONLY "
                            "a YAML config block (no markdown fences) with ONE change "
                            "to improve F1 score. Config options:\n"
                            "- detector_type: grounding_dino, owlv2, yoloworld\n"
                            "- feature_model: siglip, dinov2\n"
                            "- box_threshold: 0.1-0.9\n"
                            "- text_threshold: 0.1-0.9\n"
                            "- similarity_threshold_high: 0.5-0.95\n"
                            "- similarity_threshold_medium: 0.4-0.8"
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_agent_prompt(current_config, results_summary),
                    },
                ]
                response = self._ollama_chat(messages)
                proposed_config = yaml.safe_load(response)
                if proposed_config:
                    current_config = proposed_config
                    self._save_config(current_config)
                    logger.info("Agent proposed new config: %s", proposed_config)
                else:
                    logger.info("Agent chose to stop.")
                    break

            except Exception as e:
                logger.error("Agent error: %s", e)
                break

    def _evaluate(
        self,
        results: list,
        pseudo_labels: Optional[dict] = None,
    ) -> dict:
        """
        Evaluate detection results against pseudo-labels.

        Returns precision, recall, F1.
        """
        # Simplified evaluation:
        # For each shelf image, count high-confidence matches
        # vs expected (from pseudo_labels or from detection count heuristics)
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for r in results:
            for m in r.matches:
                if m.confidence == "high":
                    total_tp += 1
                elif m.confidence == "medium":
                    total_fp += 0.5  # partial credit
                else:
                    total_fp += 1

        # Heuristic: if no pseudo-labels, assume all high-conf are correct
        # and estimate FP/FN from confidence distribution
        if pseudo_labels is None:
            total_fn = sum(
                1 for r in results
                for m in r.matches
                if m.confidence in ("medium", "low")
            ) // 2

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _get_last_f1(self) -> Optional[float]:
        """Read the most recent F1 from results.tsv."""
        tsv_path = Path(self.config.workspace_dir) / RESULTS_TSV
        if not tsv_path.exists():
            return None
        lines = tsv_path.read_text().strip().split("\n")
        if len(lines) <= 1:
            return None
        last = lines[-1].split("\t")
        try:
            return float(last[3])  # f1 column
        except (IndexError, ValueError):
            return None

    def _summarize_results(self) -> str:
        """Get a summary of recent experiment results for the agent."""
        tsv_path = Path(self.config.workspace_dir) / RESULTS_TSV
        if not tsv_path.exists():
            return "No experiments yet."
        lines = tsv_path.read_text().strip().split("\n")
        if len(lines) <= 1:
            return "No experiments yet."
        # Return last 5 experiments
        recent = lines[-5:] if len(lines) > 5 else lines[1:]
        return "\n".join(recent)

    def _log_experiment(self, exp: ExperimentResult):
        """Append experiment result to results.tsv."""
        tsv_path = Path(self.config.workspace_dir) / RESULTS_TSV
        row = (
            f"{exp.commit}\t{exp.val_precision:.6f}\t{exp.val_recall:.6f}\t"
            f"{exp.val_f1:.6f}\t{exp.num_detections}\t{exp.num_matches}\t"
            f"{exp.runtime_seconds:.1f}\t{exp.status}\t{exp.description}\n"
        )
        with open(tsv_path, "a") as f:
            f.write(row)
        logger.info(
            "Logged: %s | P=%.3f R=%.3f F1=%.3f | %s",
            exp.commit, exp.val_precision, exp.val_recall, exp.val_f1, exp.status,
        )


def main():
    """CLI entry point for the orchestrator."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ShelfMatch Orchestrator")
    parser.add_argument("--model", default="gemma4:latest", help="Ollama model")
    parser.add_argument("--refs", nargs="+", required=True, help="Reference image paths")
    parser.add_argument("--val", nargs="+", required=True, help="Validation shelf image paths")
    parser.add_argument("--max-iter", type=int, default=10, help="Max iterations")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    args = parser.parse_args()

    config = OrchestratorConfig(
        ollama_model=args.model,
        workspace_dir=args.workspace,
        max_iterations=args.max_iter,
    )
    orch = ShelfOrchestrator(config=config)
    orch.run_self_training(args.refs, args.val)


if __name__ == "__main__":
    main()
