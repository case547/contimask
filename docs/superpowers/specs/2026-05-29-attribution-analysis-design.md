# Attribution Analysis — Design Spec

**Date:** 2026-05-29

## Context

The ContiMask paper reports per-feature *Retention* (how often each feature was kept by the learned deletion mask) and *Imbalance* (how much more likely a septic patient is to have any observation of a feature) for the patient predicted most likely to develop sepsis. We replicate the retention analysis aggregated across the top-100 predicted-septic patients, skip recomputing imbalance (already in the paper's table for PhysioNet 2019), and additionally record all test-set predicted probabilities for downstream patient-level analysis.

## Outputs

| File | Script | Contents |
|---|---|---|
| `checkpoints/test_predictions.csv` | `eval.py` | `psv_file, true_label, predicted_prob` for every test patient |
| `checkpoints/attribution_results.json` | `attribute.py` | per-feature retention rates + PSV filenames for top/bottom 100 |

## Changes to `eval.py`

Replace the single `_evaluate(model, test_loader, args.device)` call with an inline loop that:

1. Sets `model.eval()` and iterates over `test_ds` by index inside a `torch.no_grad()` block, calling `model.classify()` + `sigmoid` directly — no import from `sepsis_attribution`
2. Collects `(psv_file, true_label, predicted_prob)` per patient, keeping filenames aligned with predictions via the dataset index
3. Writes `test_predictions.csv` to `checkpoint_dir`
4. Computes and logs AUROC from the collected predictions (replacing the `_evaluate` call)

`_evaluate` in `finetune.py` is not modified. No new cross-module imports are introduced.

## Changes to `attribute.py`

### `_run_group`

Return signature changes from `(list[float], list[float])` to `(list[float], list[float], list[Tensor], list[Tensor])` — the two new lists are the per-patient `mask` and `data_mask` tensors (shape `(1, T, F)` each), collected inside the existing loop at no extra compute cost.

### New: `compute_retention`

```
compute_retention(
    masks: list[Tensor],       # each (1, T, F)
    data_masks: list[Tensor],  # each (1, T, F)
) -> Tensor  # shape (F,)
```

For each feature `f`, compute the per-patient retention rate as:

```
rate_f_patient = sum(mask[:, :, f] * data_mask[:, :, f]) / sum(data_mask[:, :, f])
```

Exclude patients where `sum(data_mask[:, :, f]) == 0` (feature never observed). Return the mean across included patients. Result is a `(39,)` tensor.

### `main()` additions

1. After `_run_group` calls, compute `retention_top = compute_retention(masks_top, dms_top)` and `retention_bot = compute_retention(masks_bot, dms_bot)`
2. Extract PSV filenames: `top_files = [test_ds.psv_files[i] for i in top_idx.tolist()]` (and similarly for `bottom_idx`)
3. Log ranked per-feature retention for the top-100 group (descending order)
4. Save `attribution_results.json`:

```json
{
  "feature_names": [...],
  "retention_top100": {"HR": 0.23, "O2Sat": 0.11, ...},
  "retention_bottom100": {"HR": 0.08, ...},
  "top100_psv_files": ["p00001.psv", ...],
  "bottom100_psv_files": ["p00042.psv", ...]
}
```

Feature names are taken from `config.FEATURE_COLS` (the ordered 39-element list already used throughout the codebase).

## Retention averaging

**Macro-average** — each patient contributes equally regardless of sequence length. Patients with zero observations for a given feature are excluded from that feature's average. This matches the proportional (10% target area) nature of the ContiMask constraint.

## What is NOT changing

- `sepsis_attribution.py` — no changes
- `training/finetune.py` — no changes
- `_evaluate` is not called from `eval.py` anymore but remains available for use during training

## Verification

1. Run `uv run eval.py` → `test_predictions.csv` exists, row count equals test set size, AUROC logged matches previous eval run
2. Run `uv run attribute.py` (on a small subset if needed) → `attribution_results.json` exists, retention values are in [0, 1] per feature, top-100 filenames are plausible PSV paths
3. Load `test_predictions.csv` in pandas, confirm `predicted_prob` is in [0, 1] and true labels match expected class balance
