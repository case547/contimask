# Mask Subsampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `subsample_mask` to `sepsis_attribution.py` and call it inside `run_attribution` so the returned binary mask covers exactly 10% of observed entries, matching the paper's post-hoc random-subsampling step.

**Architecture:** A standalone function `subsample_mask(mask, data_mask, target_area, seed)` handles the subsampling logic; `run_attribution` gains a `seed` parameter and calls `subsample_mask` on the extracted mask before returning. All existing callers (`attribute.py`) receive the correctly-subsampled mask with no changes.

**Tech Stack:** Python, PyTorch (`torch.Generator`, `torch.randperm`)

---

## File Structure

| File | Change |
|---|---|
| `sepsis_attribution.py` | Add `subsample_mask` function; add `seed` param to `run_attribution`; call `subsample_mask` at end of `run_attribution` |
| `tests/attribution/__init__.py` | Create (empty) |
| `tests/attribution/test_subsample_mask.py` | Create with all subsample tests |

---

### Task 1: Create test file for `subsample_mask`

**Files:**
- Create: `tests/attribution/__init__.py`
- Create: `tests/attribution/test_subsample_mask.py`

- [ ] **Step 1: Create the test package**

```bash
touch /home/justin/msc_ai/individual-project/contimask/tests/attribution/__init__.py
```

- [ ] **Step 2: Write the test file**

```python
# tests/attribution/test_subsample_mask.py
import torch

from sepsis_attribution import subsample_mask


def test_excess_trimmed():
    """Active observed entries are reduced to exactly n_target."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)        # all 20 entries active
    data_mask = torch.ones(1, T, F)   # all 20 entries observed
    # target_area=0.5 → n_target = round(0.5 * 20) = 10
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert int((out * data_mask).sum().item()) == 10


def test_already_at_target():
    """Mask with exactly n_target active observed entries is returned unchanged."""
    T, F = 4, 5
    data_mask = torch.ones(1, T, F)
    n_target = round(0.5 * T * F)    # 10
    mask = torch.zeros(1, T, F)
    mask.view(1, -1)[0, :n_target] = 1.0
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert torch.equal(out, mask)


def test_below_target():
    """Mask with fewer active entries than n_target is returned unchanged."""
    T, F = 4, 5
    data_mask = torch.ones(1, T, F)
    mask = torch.zeros(1, T, F)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 1] = 1.0             # only 2 active, target is 10
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert torch.equal(out, mask)


def test_reproducibility():
    """Same seed produces identical output on repeated calls."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)
    data_mask = torch.ones(1, T, F)
    out1 = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    out2 = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    assert torch.equal(out1, out2)


def test_different_seeds_give_different_outputs():
    """Different seeds (with high probability) produce different outputs."""
    T, F = 4, 5
    mask = torch.ones(1, T, F)
    data_mask = torch.ones(1, T, F)
    out1 = subsample_mask(mask, data_mask, target_area=0.5, seed=0)
    out2 = subsample_mask(mask, data_mask, target_area=0.5, seed=1)
    assert not torch.equal(out1, out2)


def test_unobserved_positions_zeroed():
    """mask=1 positions where data_mask=0 become 0 after subsampling."""
    T, F = 4, 5
    data_mask = torch.zeros(1, T, F)
    data_mask[0, :2, :] = 1.0       # only 10 entries observed, n_target=5
    mask = torch.ones(1, T, F)      # mask=1 everywhere including unobserved
    out = subsample_mask(mask, data_mask, target_area=0.5, seed=42)
    # No output entry should be 1 where data_mask is 0
    assert (out * (1.0 - data_mask)).sum().item() == 0.0


def test_batch_independence():
    """Result for item 0 in a batch-of-2 equals result for item 0 run alone."""
    T, F = 4, 5
    mask0 = torch.ones(1, T, F)
    dm0 = torch.ones(1, T, F)
    mask1 = torch.zeros(1, T, F)
    mask1[0, :2, :] = 1.0
    dm1 = torch.ones(1, T, F)

    mask_batch = torch.cat([mask0, mask1], dim=0)   # (2, T, F)
    dm_batch = torch.cat([dm0, dm1], dim=0)
    out_batch = subsample_mask(mask_batch, dm_batch, target_area=0.5, seed=42)
    out_alone = subsample_mask(mask0, dm0, target_area=0.5, seed=42)

    assert torch.equal(out_batch[0], out_alone[0])
```

- [ ] **Step 3: Run tests to confirm they all fail with ImportError**

```bash
cd /home/justin/msc_ai/individual-project/contimask && python -m pytest tests/attribution/test_subsample_mask.py -v
```

Expected: all 7 tests fail with `ImportError: cannot import name 'subsample_mask'`

---

### Task 2: Implement `subsample_mask`

**Files:**
- Modify: `sepsis_attribution.py` (add function before `run_attribution`)

- [ ] **Step 1: Add `subsample_mask` to `sepsis_attribution.py`**

Insert the following block immediately before the `def run_attribution(` line:

```python
def subsample_mask(
    mask: torch.Tensor,
    data_mask: torch.Tensor,
    target_area: float = config.MASK_TARGET_AREA,
    seed: int | None = 42,
) -> torch.Tensor:
    """Randomly subsample active mask entries to exactly target_area of observed entries.

    If the mask already covers <= target_area of observed entries, returns a clone
    unchanged.  Positions where data_mask=0 are always 0 in the output for items
    that go through the subsample path.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    B, T, F = mask.shape
    out = mask.clone()

    for b in range(B):
        n_observed = int(data_mask[b].sum().item())
        n_target = round(target_area * n_observed)

        active_indices = (
            (mask[b] * data_mask[b]).view(-1).nonzero(as_tuple=False).squeeze(1)
        )
        n_active = len(active_indices)

        if n_active <= n_target:
            continue

        perm = torch.randperm(n_active, generator=gen)
        keep = active_indices[perm[:n_target]]

        flat = torch.zeros(T * F, dtype=mask.dtype)
        flat[keep] = 1.0
        out[b] = flat.view(T, F)

    return out

```

- [ ] **Step 2: Run tests to confirm they all pass**

```bash
cd /home/justin/msc_ai/individual-project/contimask && python -m pytest tests/attribution/test_subsample_mask.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/attribution/__init__.py tests/attribution/test_subsample_mask.py sepsis_attribution.py
git commit -m "feat: add subsample_mask — post-hoc random subsampling to target area"
```

---

### Task 3: Wire `subsample_mask` into `run_attribution`

**Files:**
- Modify: `sepsis_attribution.py` (update `run_attribution` signature and final block)

- [ ] **Step 1: Add `seed` parameter and subsampling call to `run_attribution`**

Replace the `run_attribution` signature and docstring `Args` block. The new signature adds `seed: int | None = 42` as the last parameter before `device`:

```python
def run_attribution(
    model: nn.Module,
    t: torch.Tensor,
    X: torch.Tensor,
    data_mask: torch.Tensor,
    n_epoch: int = config.MASK_EPOCHS,
    K: int = 10,
    lr: float = 0.01,
    lambda_l1: float = 0.01,
    lambda_tv: float = 1.0,
    target_area: float = config.MASK_TARGET_AREA,
    mask_hidden_dim: int = config.MASK_HIDDEN_DIM,
    mask_L: int = config.MASK_L,
    n_features: int = 39,
    seed: int | None = 42,
    device: str = "cpu",
) -> tuple[ContiMask, torch.Tensor]:
```

Add `seed` to the docstring Args:
```
        seed: Seed for the post-hoc random subsampling step; ``None`` for
            non-deterministic behaviour.
```

- [ ] **Step 2: Replace the final block of `run_attribution`**

Find this block (currently the last 4 lines of `run_attribution`):
```python
    with torch.no_grad():
        mask_values = (explainer.pert_mask(t.to(device)) > 0.5).float().cpu()

    return explainer, mask_values
```

Replace with:
```python
    with torch.no_grad():
        mask_values = (explainer.pert_mask(t.to(device)) > 0.5).float().cpu()
    mask_values = subsample_mask(
        mask_values, data_mask.cpu(), target_area=target_area, seed=seed
    )
    return explainer, mask_values
```

- [ ] **Step 3: Run all attribution tests to confirm nothing is broken**

```bash
cd /home/justin/msc_ai/individual-project/contimask && python -m pytest tests/attribution/ tests/test_attribution.py -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add sepsis_attribution.py
git commit -m "feat: wire subsample_mask into run_attribution"
```
