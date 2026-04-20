# ShelfMatch 🛒

**Zero-label retail shelf product detection** using open-source foundation models.

Given reference product images (multiple angles supported) and shelf photos, detect where each product appears on the shelves — no manual bounding box annotation required.

---

## Architecture

```
Reference Images
      ↓
FeatureExtractor (SigLIP / DINOv2)
      ↓
FAISS Vector Index
      ↓                Shelf Image
      ←── Grounding DINO (detect) ──→ SAM2 (segment)
                    ↓
             Crop & Match
                    ↓
      [BBox + Mask + Product ID]
                    ↓
      Ollama / Gemma4 (orchestrator — autoresearch loop)
```

## Models Used

| Component | Model | Why |
|-----------|-------|-----|
| Detection | **Grounding DINO** | Best zero-shot, text+image prompts |
| Detection (fast) | **YOLO-World** | Real-time, fine-tunable |
| Segmentation | **SAM2** | State-of-the-art segmentation |
| Feature Embedding | **SigLIP** | Strong product matching (contrastive) |
| Feature Embedding | **DINOv2** | Fine-grained visual similarity |
| Vector Search | **FAISS** | Fast similarity at scale |
| Orchestration | **Gemma4 (Ollama)** | Local LLM drives the loop |

## Quick Start

### 1. Install

```bash
cd ~/Projects/shelfmatch
uv sync

# SAM2 (Segment Anything 2)
pip install git+https://github.com/facebookresearch/sam2

# HQ-SAM (optional, better mask quality)
pip install git+https://github.com/SysCV/sam-hq

# Ollama (for orchestrator)
ollama pull gemma4:latest
```

### 2. Use as Python API

```python
from shelfmatch import ShelfMatcher

matcher = ShelfMatcher()
matcher.load_references(["product_a.jpg", "product_b.jpg"])
results = matcher.detect("shelf.jpg")

for m in results.matches:
    print(f"{m.product_name}: {m.confidence} (sim={m.similarity:.3f})")
```

### 3. Web App

```bash
cd ~/Projects/shelfmatch
uv run python -m shelfmatch.webapp.main
# → http://localhost:7860
```

Upload reference products on the left, shelf image, click **Detect Products**.

---

## Project Structure

```
shelfmatch/
├── program.md                        ← Agent instructions (autoresearch)
├── pyproject.toml
├── src/shelfmatch/
│   ├── pipeline/
│   │   ├── detector.py              ← Grounding DINO, OWLv2, YOLO-World
│   │   ├── segmenter.py             ← SAM2, HQ-SAM
│   │   ├── matcher.py               ← SigLIP/DINOv2 + FAISS matching
│   │   └── shelfmatcher.py         ← Unified pipeline
│   ├── orchestrator/
│   │   └── main.py                  ← Ollama autoresearch loop
│   ├── training/
│   │   └── synthetic.py            ← Synthetic shelf generation
│   ├── webapp/
│   │   └── main.py                 ← Gradio web app
│   └── cli.py                       ← CLI tools
├── workspace/
│   ├── config/
│   ├── data/
│   └── archive/
└── references/
```

---

## CLI Commands

### Detect products
```bash
uv run shelfmatch detect \
  -r product_a.jpg product_b.jpg \
  -s shelf.jpg \
  -d grounding_dino \
  -f siglip
```

### Generate synthetic training data
```bash
uv run shelfmatch synthesize \
  -r product_a.jpg product_b.jpg \
  -o workspace/data/synthetic \
  -n 200 --min-products 3 --max-products 8
```

---

## Near-Zero Labeling Strategy

The key insight: **you don't need to label a single bounding box.**

1. **Your reference images ARE the labels.** Load them as reference products.
2. **Run the pipeline** — high-confidence matches become pseudo-labels.
3. **Synthetic data** — composite your reference products onto shelf backgrounds with auto-annotations.
4. **Self-training loop** — Gemma4 via Ollama iterates on thresholds/config automatically (autoresearch pattern from Andrej Karpathy).

---

## Autoresearch Loop

The `orchestrator/` module applies Karpathy's `autoresearch` pattern:

```bash
uv run python -m shelfmatch.orchestrator.main \
  --refs product_a.jpg product_b.jpg \
  --val val_shelf_1.jpg val_shelf_2.jpg \
  --model gemma4:latest \
  --max-iter 10
```

The agent:
1. Reads `program.md` for instructions
2. Modifies `config.yaml` (thresholds, model choices)
3. Runs the pipeline on validation data
4. Evaluates F1 — keeps or discards changes
5. Logs to `results.tsv`
6. Proposes the next experiment

---

## Setup Notes

- **SAM2 checkpoints**: Download from https://github.com/facebookresearch/sam2#model-checkpoints and place in `~/.cache/sam2/`
- **Ollama**: Install from https://ollama.com. Pull models you want to use.
- **GPU recommended** for detection (CUDA). Runs on CPU with reduced performance.
- **References**: Use multiple angles per product for best matching robustness.
