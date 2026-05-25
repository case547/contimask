# Mask Subsampling Implementation Design

## Goal

After ContiMask attribution, the returned binary mask typically covers more than the target 10% of observed entries. Following the paper, randomly subsample the active positions down to exactly 10% of observed entries per patient.

## Architecture

A standalone function `subsample_mask` in `sepsis_attribution.py`, called inside `run_attribution` immediately after the mask is extracted. Callers of `run_attribution` (i.e. `attribute.py`) receive the already-subsampled mask with no changes required on their side.

## `subsample_mask` function

**Location:** `sepsis_attribution.py`

**Signature:**
```python
def subsample_mask(
    mask: torch.Tensor,
    data_mask: torch.Tensor,
    target_area: float = config.MASK_TARGET_AREA,
    seed: int | None = 42,
) -> torch.Tensor:
```

**Parameters:**
- `mask`: `(B, T, F)` binary float tensor — the raw thresholded mask from `run_attribution`; must be on CPU
- `data_mask`: `(B, T, F)` binary float tensor — the observation mask for the same patient(s); must be on CPU
- `target_area`: fraction of observed entries to keep; defaults to `config.MASK_TARGET_AREA` (0.1)
- `seed`: integer seed for `torch.Generator`; `None` for non-deterministic behaviour; defaults to `0`

Both tensors must be on CPU. `torch.Generator` is device-specific and this function does not support CUDA tensors.

**Logic:**

Create one `torch.Generator()`, seeded once before the batch loop if `seed` is not `None`.

Per batch item `b`:
1. `n_observed = int(data_mask[b].sum())`
2. `n_target = round(target_area * n_observed)`
3. `active_indices` = flat indices where `mask[b]=1` AND `data_mask[b]=1`
4. If `len(active_indices) <= n_target`: copy item `b` from `mask` unchanged and continue
5. Otherwise:
   - `perm = torch.randperm(len(active_indices), generator=gen)`
   - `keep = active_indices[perm[:n_target]]`
   - Build a zero tensor of shape `(T*F,)`, set `keep` positions to 1, reshape to `(T, F)`

Building from scratch (rather than zeroing in-place) also cleans up any spurious `mask=1 & data_mask=0` positions.

**Returns:** `torch.Tensor` of the same shape and dtype as `mask`, on CPU.

## Changes to `run_attribution`

Add parameter `seed: int | None = 0`, passed through to `subsample_mask`.

Replace the existing final block:
```python
# before
with torch.no_grad():
    mask_values = (explainer.pert_mask(t.to(device)) > 0.5).float().cpu()
return explainer, mask_values
```
with:
```python
# after
with torch.no_grad():
    mask_values = (explainer.pert_mask(t.to(device)) > 0.5).float().cpu()
mask_values = subsample_mask(mask_values, data_mask.cpu(), target_area=target_area, seed=seed)
return explainer, mask_values
```

No other call sites change.

## Testing

New test file: `tests/attribution/test_subsample_mask.py`

| Test | What it checks |
|---|---|
| Already at target | mask with exactly `n_target` active observed entries is returned unchanged |
| Below target | mask with fewer than `n_target` active observed entries is returned unchanged |
| Excess trimmed | output has exactly `n_target` active observed entries |
| Reproducibility | same seed → identical output on two calls |
| Different seeds | two different seeds → different outputs (with high probability) |
| Unobserved positions zeroed | any `mask=1 & data_mask=0` positions become 0 |
| Batch independence | result for item 0 in a batch of 2 equals result for item 0 run alone with same seed |
