# Sepsis Prediction: Training and Attribution

End-to-end guide for training the DiffusionTransformer on the PhysioNet 2019 sepsis dataset and running ContiMask attribution.

---

## Prerequisites

**Data:** PhysioNet Challenge 2019 training set A, extracted to a local directory. The default path assumed throughout is:

```
~/msc_ai/individual-project/sepsis_data/training_setA/
```

The directory should contain one `.psv` file per patient (pipe-separated, one row per ICU hour).

**Environment:** All commands use `uv run` to ensure the correct virtual environment. Run from the project root.

---

## Step 1 — Train

```bash
uv run python training/train.py \
  --phase both \
  --data_dir ~/msc_ai/individual-project/sepsis_data/training_setA \
  --checkpoint_dir checkpoints
```

`--phase` accepts `pretrain`, `finetune`, or `both` (default).

- **Pretrain** corrupts observed features with DDPM noise and trains the transformer to denoise them. Saves `checkpoints/pretrained_backbone.pt`.
- **Finetune** attaches a classification head and trains on binary sepsis labels with BCE loss and per-layer learning rate ratios. Saves `checkpoints/best_model.pt` (best validation AUROC) and `checkpoints/norm_stats.pt` (per-feature z-score statistics computed from the training split).

Both phases use early stopping (patience 10) on the validation set. If `checkpoints/norm_stats.pt` already exists it is reused, skipping the ~20 s recomputation.

To run phases separately:

```bash
# Phase 1 only
uv run python training/train.py --phase pretrain

# Phase 2 only (loads pretrained_backbone.pt if present)
uv run python training/train.py --phase finetune
```

Key hyperparameters are in `config.py`. The data is split 70 / 15 / 15 (train / val / test) using a stratified split with a fixed seed (42), so the split is identical across all runs.

---

## Step 2 — Evaluate

```bash
uv run python eval.py \
  --checkpoint_dir checkpoints \
  --data_dir ~/msc_ai/individual-project/sepsis_data/training_setA
```

Recovers the held-out test split (same seed), loads `best_model.pt` and `norm_stats.pt`, and prints the test-set AUROC:

```
Test AUROC: 0.XXXX
```

---

## Step 3 — Attribution

```bash
uv run python attribute.py \
  --checkpoint_dir checkpoints \
  --data_dir ~/msc_ai/individual-project/sepsis_data/training_setA
```

Runs ContiMask saliency attribution and reports odds-change metrics. The process:

1. Scores all test patients with the trained model.
2. Selects the 100 patients with the **highest** predicted sepsis probability and the 100 with the **lowest**.
3. For each of the 200 patients, trains an instance-specific saliency mask using ContiMask (Deletion perturbation, NeuroEvolution optimisation, 200 epochs, 10 % target area). Each mask identifies the 10 % of observed data points most salient for that patient's prediction.
4. Computes two odds-change metrics per patient:
   - **Del odds change** — applies the mask as a deletion (zeros both feature values and the observation mask where the mask is 0) and measures the mean change in log-odds.
   - **Imp odds change** — imputes 0 (≈ feature mean in normalised space) for non-salient features while leaving the observation mask unchanged, then measures the mean change in log-odds.

Per-patient progress is printed as attribution runs. Final output:

```
=== Odds-change results ===
Del odds change  |  top 100:    +X.XXXX
Del odds change  |  bottom 100: +X.XXXX
Imp odds change  |  top 100:    +X.XXXX
Imp odds change  |  bottom 100: +X.XXXX
```

Attribution is slow — each of the 200 patients requires training a new mask network for 200 epochs. Expect several hours on CPU; a GPU significantly reduces this.

---

## Output files

| File | Created by | Contents |
|---|---|---|
| `checkpoints/pretrained_backbone.pt` | `train.py --phase pretrain` | Transformer weights after DDPM pretraining |
| `checkpoints/best_model.pt` | `train.py --phase finetune` | Best classification checkpoint (by val AUROC) |
| `checkpoints/norm_stats.pt` | `train.py` | Per-feature mean and std for z-score normalisation |
