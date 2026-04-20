"""
SigLIP contrastive fine-tuning using LoRA.

Self-training loop:
1. Generate synthetic shelf images with known product placements (SyntheticShelfGenerator)
2. Extract product crops and build contrastive pairs (TrainingFormatter)
3. Fine-tune SigLIP vision encoder with LoRA using contrastive loss
4. The fine-tuned model improves feature matching on real shelf images

Contrastive loss: InfoNCE / SimCLR-style
- Positive pair: same product_id → high similarity
- Negative pairs: different product_ids → low similarity
- Loss = -log(sigmoid(pos_sim)) - log(1 - sigmoid(neg_sim))

Usage:
    python -m shelfmatch.training.finetune \
        --pairs workspace/data/synthetic/contrastive_pairs.jsonl \
        --output workspace/models/siglip-lora \
        --epochs 5 \
        --lr 1e-4 \
        --rank 16
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Set threading env vars BEFORE importing torch
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Ensure project src is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shelfmatch.pipeline.matcher import FeatureExtractor


class ContrastiveDataset(Dataset):
    """
    Dataset that yields (anchor_img, positive_img, [negative_imgs], anchor_label).

    Loads images from disk paths stored in the JSONL.
    """

    def __init__(self, pairs_path: str | Path, image_size: int = 224):
        self.pairs_path = Path(pairs_path)
        self.image_size = image_size
        self.examples: list[dict] = []
        self._load()

    def _load(self):
        with open(self.pairs_path) as f:
            for line in f:
                self.examples.append(json.loads(line))
        logger.info("Loaded %d contrastive pairs from %s", len(self.examples), self.pairs_path)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]

        anchor = self._load_image(ex["anchor_path"])
        positive = self._load_image(ex["positive_path"])
        negatives = [self._load_image(p) for p in ex["negative_paths"]]

        return {
            "anchor": anchor,
            "positive": positive,
            "negatives": negatives,
            "anchor_label": ex["anchor_label"],
        }

    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            import numpy as np
            arr = np.array(img, dtype=np.float32) / 255.0
            # Normalize with ImageNet stats (SigLIP uses this)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
            tensor = (tensor - mean) / std
            return tensor
        except Exception as e:
            # Return a blank image on error
            return torch.zeros(3, self.image_size, self.image_size)


class SigLIPContrastiveLoss(nn.Module):
    """
    SigLIP-style contrastive loss.

    Instead of softmax over all negatives in batch, we use local negatives per sample.
    This makes it tractable with a fixed negative set per example.

    Loss = -log(sigmoid(pos_sim)) - sum_i log(1 - sigmoid(neg_sim_i))

    where sim = temperature * cos(emb_a, emb_p/emb_n)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        # Learnable temperature (as in SigLIP paper)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / temperature))

    def forward(
        self,
        anchor_emb: torch.Tensor,       # [B, D]
        positive_emb: torch.Tensor,      # [B, D]
        negative_embs: list[torch.Tensor]  # list of [B, D]
    ) -> torch.Tensor:
        """
        Args:
            anchor_emb: [B, D] anchor embeddings
            positive_emb: [B, D] positive embeddings
            negative_embs: list of [B, D], each is a different negative set
        """
        B = anchor_emb.size(0)

        # Normalize embeddings
        anchor_emb = torch.nn.functional.normalize(anchor_emb, p=2, dim=1)
        positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)

        # Positive similarity
        pos_sim = torch.sum(anchor_emb * positive_emb, dim=1)  # [B]
        pos_sim = pos_sim * torch.exp(self.logit_scale)

        # Negative similarities — collect all negatives
        all_neg_emb = torch.cat(negative_embs, dim=0)  # [num_neg, D]
        all_neg_emb = torch.nn.functional.normalize(all_neg_emb, p=2, dim=1)

        # For each anchor, compute similarity to all negatives
        # anchor_emb: [B, 1, D], all_neg_emb: [1, num_neg, D] → [B, num_neg]
        neg_sim = torch.einsum("bd,nd->bn", anchor_emb, all_neg_emb)
        neg_sim = neg_sim * torch.exp(self.logit_scale)

        # SigLIP loss
        # loss = -log(sigmoid(pos_sim)) - sum_i log(1 - sigmoid(neg_sim_i))
        pos_loss = -torch.nn.functional.logsigmoid(pos_sim).mean()
        neg_loss = -torch.nn.functional.logsigmoid(-neg_sim).mean()

        return pos_loss + neg_loss

    def get_scale(self) -> float:
        return float(torch.exp(self.logit_scale).detach().cpu().item())


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate to handle variable-length negatives."""
    anchors = torch.stack([b["anchor"] for b in batch])
    positives = torch.stack([b["positive"] for b in batch])
    max_neg = max(len(b["negatives"]) for b in batch)
    negatives = torch.zeros(len(batch), max_neg, batch[0]["anchor"].shape[0],
                            batch[0]["anchor"].shape[1], batch[0]["anchor"].shape[2])
    for i, b in enumerate(batch):
        for j, n in enumerate(b["negatives"]):
            negatives[i, j] = n
    return {
        "anchor": anchors,
        "positive": positives,
        "negatives": negatives,
    }


def train(
    pairs_path: str,
    output_dir: str,
    model_name: str = "SigLIP-ViT-SO400M/siglip-so400m-patch14-224",
    epochs: int = 5,
    lr: float = 1e-4,
    rank: int = 16,
    alpha: int = 32,
    batch_size: int = 8,
    image_size: int = 224,
    warmup_steps: int = 100,
    max_steps: Optional[int] = None,
    seed: int = 42,
):
    """
    Fine-tune SigLIP with LoRA on contrastive pairs.

    Args:
        pairs_path: Path to contrastive pairs JSONL
        output_dir: Where to save LoRA weights
        model_name: HuggingFace model ID for SigLIP
        epochs: Number of training epochs
        lr: Learning rate
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        batch_size: Per-device batch size
        image_size: Input image size
        warmup_steps: Warmup steps for LR schedule
        max_steps: Override epochs with fixed step count
        seed: Random seed
    """
    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model and tokenizer
    logger.info("Loading SigLIP model: %s", model_name)
    extractor = FeatureExtractor(model_name=model_name)
    model = extractor.model
    processor = extractor.processor
    model.to(device)
    model.train()

    # Set up LoRA
    from peft import LoraConfig, get_peft_model, TaskType

    # SigLIP ViT-Base has 12 encoder layers
    NUM_LAYERS = 12
    layers = []
    for i in range(NUM_LAYERS):
        layers.extend([
            f"vision_model.encoder.layers.{i}.self_attn.q_proj",
            f"vision_model.encoder.layers.{i}.self_attn.v_proj",
            f"vision_model.encoder.layers.{i}.mlp.fc1",
            f"vision_model.encoder.layers.{i}.mlp.fc2",
        ])

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=layers,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset and dataloader
    dataset = ContrastiveDataset(pairs_path, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max_steps or (len(dataloader) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Loss
    criterion = SigLIPContrastiveLoss(temperature=0.1).to(device)

    logger.info("Starting training: %d steps, batch_size=%d", total_steps, batch_size)
    step = 0
    for epoch in range(epochs):
        if max_steps and step >= max_steps:
            break

        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            anchors = batch["anchor"].to(device)
            positives = batch["positive"].to(device)
            negatives = batch["negatives"].to(device)

            # Extract embeddings
            anchor_emb = model(anchors)
            positive_emb = model(positives)

            # Flatten negatives: [B, num_neg, C, H, W] -> list of [B, C, H, W]
            B, num_neg = negatives.shape[:2]
            neg_list = [negatives[:, i] for i in range(num_neg)]
            neg_embs = [model(n) for n in neg_list]

            # Loss and backprop
            optimizer.zero_grad()
            loss = criterion(anchor_emb, positive_emb, neg_embs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "scale": f"{criterion.get_scale():.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if max_steps and step >= max_steps:
                break

        avg_loss = epoch_loss / len(dataloader)
        logger.info(
            "Epoch %d/%d — avg_loss=%.4f, scale=%.2f",
            epoch + 1, epochs, avg_loss, criterion.get_scale(),
        )

        # Save checkpoint
        checkpoint_path = output_dir / f"epoch-{epoch+1:02d}"
        model.save_pretrained(str(checkpoint_path))
        logger.info("Saved checkpoint: %s", checkpoint_path)

    # Save final model
    final_path = output_dir / "final"
    model.save_pretrained(str(final_path))
    logger.info("Training complete. Final model: %s", final_path)

    return final_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune SigLIP with LoRA")
    parser.add_argument("--pairs", required=True, help="Contrastive pairs JSONL")
    parser.add_argument("--output", required=True, help="Output directory for LoRA weights")
    parser.add_argument("--model", default="SigLIP-ViT-SO400M/siglip-so400m-patch14-224",
                        help="SigLIP model name")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        pairs_path=args.pairs,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        batch_size=args.batch_size,
        image_size=args.image_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
