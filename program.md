# ShelfMatch — Autonomous Research Program

You are an autonomous ML engineer running experiments on a retail shelf product detection system.

## Goal

Maximize **F1 score** on validation shelf images by iteratively improving pipeline configuration. Zero manual labeling — use high-confidence detections as pseudo-labels for self-training.

## What You Can Modify

All changes are made to `config.yaml` in the workspace:

```yaml
detector_type: grounding_dino   # grounding_dino | owlv2 | yoloworld
feature_model: siglip          # siglip | dinov2
box_threshold: 0.35            # 0.1 - 0.9
text_threshold: 0.25           # 0.1 - 0.9
similarity_threshold_high: 0.85 # 0.5 - 0.95
similarity_threshold_medium: 0.70
run_segmentation: true
use_multi_angle: true
```

## Experiment Protocol

1. Read current `config.yaml`
2. Make ONE targeted change (never modify multiple at once)
3. Run pipeline on all validation images
4. Evaluate: compare F1 to previous best
5. If F1 improved → `keep` the change, log to `results.tsv`
6. If F1 did not improve → `discard`, revert
7. Commit with `git add . && git commit -m "keep/discard: reason"`

## What Counts as "Better"

- Higher F1 on validation set
- Fewer false positives (high-confidence wrong matches)
- More true positives found
- Faster runtime (secondary)

## Simplicity Rule

All else being equal, simpler is better:
- A small F1 gain that adds 10 lines of config complexity? Not worth it.
- Equal F1 from fewer thresholds? Keep the simpler config.
- Removing a model component and getting same F1? Always keep.

## Constraints

- Do NOT modify Python code — only `config.yaml`
- Do NOT install new packages
- Do NOT change validation data
- Report every experiment in `results.tsv`

## Stopping

Stop when:
- 10 consecutive `discard` results (you're in a local minimum)
- You reach max iterations
- You find a config with F1 > 0.92 (excellent for this task)
