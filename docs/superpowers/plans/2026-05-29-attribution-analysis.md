# Attribution Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-feature retention rates from ContiMask attribution, PSV filenames for top/bottom 100 predicted-septic patients, and all test-set predicted probabilities.

**Architecture:** `compute_retention` is a pure function added to `attribute.py` that macro-averages per-feature mask retention across patients; `_run_group` is extended to return the masks it already computes; `main()` calls these and saves two output files. `eval.py` gains an inline prediction loop that saves a CSV alongside the existing AUROC log.

**Tech Stack:** PyTorch, Python `json` + `csv` stdlib, scikit-learn (already present), `uv run` for execution.

---

## File Map

| File | Change |
|---|---|
| `attribute.py` | Add `compute_retention`; extend `_run_group` return; update `main()` |
| `eval.py` | Replace `_evaluate` call with inline prediction loop; save CSV |
| `tests/attribution/test_retention.py` | New — unit tests for `compute_retention` |

---

### Task 1: `compute_retention` — tests first

**Files:**
- Create: `tests/attribution/test_retention.py`
- Modify: `attribute.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/attribution/test_retention.py`:

```python
import torch

from attribute import compute_retention


def test_macro_average():
    # Patient 0: feature 0 observed 4 times, retained 2 → rate 0.5
    # Patient 1: feature 0 observed 2 times, retained 2 → rate 1.0
    # Macro-average for feature 0: (0.5 + 1.0) / 2 = 0.75
    masks = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    masks[0][0, :2, 0] = 1.0
    masks[1][0, :2, 0] = 1.0

    dms = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    dms[0][0, :, 0] = 1.0   # patient 0: 4 observed
    dms[1][0, :2, 0] = 1.0  # patient 1: 2 observed

    retention = compute_retention(masks, dms)

    assert retention.shape == (2,)
    assert abs(retention[0].item() - 0.75) < 1e-5
    assert retention[1].item() == 0.0  # feature 1 never observed


def test_excludes_unobserved_patients():
    # Patient 0: feature 1 NOT observed → excluded from feature 1 average
    # Patient 1: feature 1 observed 2 times, retained 1 → rate 0.5
    masks = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    masks[1][0, 0, 1] = 1.0

    dms = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    dms[1][0, :2, 1] = 1.0

    retention = compute_retention(masks, dms)

    assert abs(retention[1].item() - 0.5) < 1e-5


def test_zero_retention():
    # Masks are all zero — nothing retained
    masks = [torch.zeros(1, 4, 3)]
    dms = [torch.ones(1, 4, 3)]

    retention = compute_retention(masks, dms)

    assert torch.equal(retention, torch.zeros(3))


def test_full_retention():
    # Masks equal data_mask — everything retained
    dm = torch.zeros(1, 4, 3)
    dm[0, :2, :] = 1.0  # 2 timesteps observed
    masks = [dm.clone()]
    dms = [dm.clone()]

    retention = compute_retention(masks, dms)

    assert torch.allclose(retention, torch.ones(3))


def test_no_observations_anywhere():
    # data_mask is all zeros → all features return 0.0, no division by zero
    masks = [torch.ones(1, 4, 2)]
    dms = [torch.zeros(1, 4, 2)]

    retention = compute_retention(masks, dms)

    assert torch.equal(retention, torch.zeros(2))
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/attribution/test_retention.py -v
```

Expected: `ImportError` or `AttributeError` — `compute_retention` does not exist yet.

- [ ] **Step 3: Implement `compute_retention` in `attribute.py`**

Add `import json` to the imports at the top of `attribute.py` (needed for Task 2). Then add this function after the `_score_all` function (before `_run_group`):

```python
def compute_retention(
    masks: list[torch.Tensor],
    data_masks: list[torch.Tensor],
) -> torch.Tensor:
    F = masks[0].shape[-1]
    totals = torch.zeros(F)
    counts = torch.zeros(F)
    for mask, dm in zip(masks, data_masks):
        m = mask.squeeze(0)        # (T, F)
        d = dm.squeeze(0)          # (T, F)
        n_obs = d.sum(dim=0)       # (F,)
        n_ret = (m * d).sum(dim=0) # (F,)
        observed = n_obs > 0
        totals[observed] += n_ret[observed] / n_obs[observed]
        counts[observed] += 1.0
    return torch.where(counts > 0, totals / counts, torch.zeros(F))
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/attribution/test_retention.py -v
```

Expected: 5 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add tests/attribution/test_retention.py attribute.py
git commit -m "feat: add compute_retention for per-feature macro-averaged mask retention"
```

---

### Task 2: Extend `_run_group` and update `main()` in `attribute.py`

**Files:**
- Modify: `attribute.py`

- [ ] **Step 1: Update `_run_group` to return masks and data_masks**

Replace the existing `_run_group` function with:

```python
def _run_group(
    name: str,
    indices: torch.Tensor,
    probs: torch.Tensor,
    test_ds: SepsisDataset,
    model: torch.nn.Module,
    device: str,
) -> tuple[list[float], list[float], list[torch.Tensor], list[torch.Tensor]]:
    del_vals, imp_vals = [], []
    masks_list, dms_list = [], []
    logger.info("Running attribution for %s...", name)
    t0 = time.perf_counter()

    for rank, idx in enumerate(indices.tolist()):
        t, X, dm, _ = test_ds[idx]
        t, X, dm = t.unsqueeze(0), X.unsqueeze(0), dm.unsqueeze(0)

        t1 = time.perf_counter()
        _, mask = run_attribution(model, t, X, dm, device=device)
        elapsed = time.perf_counter() - t1

        del_val = compute_del_odds_change(model, t, X, dm, mask, device=device).item()
        imp_val = compute_imp_odds_change(model, t, X, dm, mask, device=device).item()

        del_vals.append(del_val)
        imp_vals.append(imp_val)
        masks_list.append(mask.cpu())
        dms_list.append(dm.cpu())
        logger.info(
            "  [%3d/100] prob=%.3f  del=%+.4f  imp=%+.4f  (%.0fs)",
            rank + 1,
            probs[idx],
            del_val,
            imp_val,
            elapsed,
        )

    total_elapsed = time.perf_counter() - t0
    logger.info("Finished %s attribution. Total time: %.1fs", name, total_elapsed)

    return del_vals, imp_vals, masks_list, dms_list
```

- [ ] **Step 2: Update `main()` to use the new return values and save `attribution_results.json`**

Replace the two `_run_group` calls and the logging block at the end of `main()` with:

```python
    del_top, imp_top, masks_top, dms_top = _run_group(
        "top 100 (most likely septic)", top_idx, probs, test_ds, model, args.device
    )
    del_bot, imp_bot, masks_bot, dms_bot = _run_group(
        "bottom 100 (least likely septic)",
        bottom_idx,
        probs,
        test_ds,
        model,
        args.device,
    )

    retention_top = compute_retention(masks_top, dms_top)
    retention_bot = compute_retention(masks_bot, dms_bot)

    top_files = [str(test_ds.psv_files[i]) for i in top_idx.tolist()]
    bot_files = [str(test_ds.psv_files[i]) for i in bottom_idx.tolist()]

    logger.info("=== Odds-change results ===")
    logger.info("Del odds change  |  top 100:    %+.4f", sum(del_top) / len(del_top))
    logger.info("Del odds change  |  bottom 100: %+.4f", sum(del_bot) / len(del_bot))
    logger.info("Imp odds change  |  top 100:    %+.4f", sum(imp_top) / len(imp_top))
    logger.info("Imp odds change  |  bottom 100: %+.4f", sum(imp_bot) / len(imp_bot))

    logger.info("=== Retention — top 100 (ranked) ===")
    ranked = sorted(
        zip(config.FEATURE_COLS, retention_top.tolist()), key=lambda x: -x[1]
    )
    for feat, rate in ranked:
        logger.info("  %-25s %.4f", feat, rate)

    results = {
        "feature_names": config.FEATURE_COLS,
        "retention_top100": dict(zip(config.FEATURE_COLS, retention_top.tolist())),
        "retention_bottom100": dict(zip(config.FEATURE_COLS, retention_bot.tolist())),
        "top100_psv_files": top_files,
        "bottom100_psv_files": bot_files,
    }
    json_path = checkpoint_dir / "attribution_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved attribution results to %s", json_path)
```

- [ ] **Step 3: Verify existing tests still pass**

```bash
uv run pytest tests/attribution/ -v
```

Expected: all tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add attribute.py
git commit -m "feat: record per-feature retention and PSV filenames in attribution_results.json"
```

---

### Task 3: Save `test_predictions.csv` from `eval.py`

**Files:**
- Modify: `eval.py`

- [ ] **Step 1: Replace imports and the prediction call in `eval.py`**

Replace the import block at the top with:

```python
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import config
from data.dataset import SepsisDataset
from data.split import stratified_patient_split
from training.train import build_model
from utils.tools import setup_logging
```

(Removes only `_evaluate`; adds `csv` and `roc_auc_score`. `DataLoader` and `config` are kept for batched inference.)

- [ ] **Step 2: Replace the `main()` body from the `_evaluate` call onwards**

Replace these lines in `main()`:

```python
    auc = _evaluate(model, test_loader, args.device)
    logger.info("Test AUROC: %.4f", auc)
```

with:

```python
    model.eval()
    all_preds, all_labels, all_files = [], [], []
    patient_idx = 0
    with torch.no_grad():
        for t, X, dm, labels in DataLoader(test_ds, batch_size=config.BATCH_SIZE):
            probs = torch.sigmoid(
                model.classify(
                    t.to(args.device), X.to(args.device), dm.to(args.device)
                ).squeeze(-1)
            ).cpu()
            for prob, label in zip(probs.tolist(), labels.tolist()):
                all_preds.append(prob)
                all_labels.append(int(label))
                all_files.append(str(test_ds.psv_files[patient_idx]))
                patient_idx += 1

    csv_path = checkpoint_dir / "test_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["psv_file", "true_label", "predicted_prob"]
        )
        writer.writeheader()
        for fname, label, prob in zip(all_files, all_labels, all_preds):
            writer.writerow(
                {"psv_file": fname, "true_label": label, "predicted_prob": prob}
            )

    auc = float(roc_auc_score(all_labels, all_preds))
    logger.info("Test AUROC: %.4f", auc)
    logger.info("Saved test predictions → %s", csv_path)
```

- [ ] **Step 3: Verify by running eval**

```bash
uv run eval.py
```

Expected output in `eval.log`:
```
Test AUROC: <matches previous run>
Saved test predictions → checkpoints/test_predictions.csv
```

- [ ] **Step 4: Spot-check the CSV**

```bash
head -5 checkpoints/test_predictions.csv
wc -l checkpoints/test_predictions.csv
```

Expected: header + one row per test patient; `predicted_prob` values between 0 and 1.

- [ ] **Step 5: Commit**

```bash
git add eval.py
git commit -m "feat: save per-patient test predictions to test_predictions.csv"
```
