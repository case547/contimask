# Sepsis DiffusionTransformer + ContiMask — Option 1 Implementation Design

**Date:** 2026-05-18
**Status:** Approved

---

## Context

The ContiMask repo currently contains the attribution framework and four synthetic benchmark experiments (Rare Feature / Rare Time, value and temp variants). None of these use a trained model — `f_to_explain` is a hand-written oracle function in every case, and the data is fully observed and regularly sampled.

The goal of this implementation is to apply ContiMask to a real clinical task: predicting whether a patient develops sepsis, using the PhysioNet 2019 Challenge dataset. This requires:

1. A trained black-box model that accepts `(t, X, data_mask)` and returns a scalar per patient
2. A data pipeline that produces the correct inputs from the raw PSV files
3. A `forward_func` wrapper compatible with ContiMask's interface (including the K>1 path used by Deletion)

The model architecture is **Option 1**: a transformer backbone pretrained with a DDPM denoising objective, then fine-tuned for binary sepsis classification. At attribution time, inference is a single forward pass — no diffusion steps.

---

## Task Definition

Following Kidger et al. and the ContiMask paper:

- **Input window:** First 72 hours of each patient's ICU stay (ICULOS ≤ 72)
- **Label:** Patient-level binary — 1 if `SepsisLabel == 1` at *any* row in the full PSV file (not just within the 72h window). A patient who develops sepsis on day 10 is labelled positive even though it is not visible in the observation window.
- **Time axis:** `t = ICULOS / 72`, normalised to `[0, 1]`
- **All patients kept** regardless of stay length — no minimum ICULOS filter, matching the NCDE paper

---

## Features

34 time-varying features (8 vital signs + 26 lab values) sampled irregularly at up to 1-hour resolution with ≈90% missing values. 5 static/demographic features (Age, Gender, Unit1, Unit2, HospAdmTime) that are constant across all rows of a patient's PSV file.

All 39 features are treated uniformly. Static features are represented as constant time series — their value is repeated at every real timestep, identical to how they appear in the PSV format. This gives ContiMask clean attribution over static features: deleting a static feature from X removes it from the model entirely, which is why the ContiMask paper (Table 5) finds age, height, and ICUType among the top-attributed features.

**Handling missing values:**
- NaN → 0 for all features (after per-feature normalisation, 0 equals the feature mean — a neutral fill)
- `data_mask`: binary tensor, shape `(B, T, 39)`. Value is 1 where a feature was actually observed, 0 where NaN-filled or padded. Static features have `data_mask = 1` at every real timestep; padded positions have `data_mask = 0` for all 39 features.

**Normalisation:** Per-feature z-score using training-set mean and std computed at dataset build time. Stats saved to `checkpoints/norm_stats.pt` and reloaded at attribution time.

**Padding:** Sequences shorter than 72 timesteps are zero-padded to length 72 with `data_mask = 0` for all padded positions.

**Data split:** 70 / 15 / 15 train/val/test, patient-level, stratified by label.

---

## Architecture: DiffusionTransformer

```
X (B, T, 39) + t (B, T)         ← all 39 features (time-varying + static as constant series)
        │
  Linear(39, d_model) + SinusoidalTimeEmbedding(t, L=64)
        │
        ▼
sequence tokens (B, T, d_model)

[CLS | sequence tokens] → (B, T+1, d_model)   ← CLS is a fixed learned nn.Parameter
        │
  + DiffusionStepEmbedding(s)    ← s=None at inference → zero contribution
        │
  TransformerEncoder(
      n_layers=4, d_model=64, n_heads=4, ffn_dim=256,
      dropout=0.1, activation=GELU,
      src_key_padding_mask  ← masks positions where data_mask[:, i, :].any(-1) == False
                               (i.e. fully padded rows beyond patient's actual stay)
  )
        │
  ┌─────┴──────┐
  │            │
CLS output   sequence outputs (B, T, d_model)
  │            │
ClassHead    DenoisingHead: Linear(d_model, 39)   ← training only, discarded at inference
  │
  MLP(d_model → 64 → 1) → logit (B, 1)
```

**CLS token:** A standard fixed learned `nn.Parameter` of shape `(1, 1, d_model)`, broadcast across the batch. Not patient-specific. The transformer learns to aggregate patient-specific information into it via self-attention.

**SinusoidalTimeEmbedding:** Encodes `t_i` as `[sin(2π·2^k·t_i), cos(2π·2^k·t_i)]` for `k=0..L-1` (L=64), then projects to `d_model`. Captures fine-grained and coarse temporal structure without relying on integer positions.

**DiffusionStepEmbedding:** Same sinusoidal scheme applied to the scalar `s`, projected to `d_model`, broadcast and added to all tokens. The model accepts `s=None`; when `None`, the step embedding returns a zero tensor. Pretraining passes a real `s`; finetuning and inference pass `s=None`.

**src_key_padding_mask:** Position `i` is excluded from attention when `data_mask[:, i, :].any(-1) == False`. Because static features have `data_mask = 1` at all real timesteps, this correctly marks only the zero-padded tail positions as masked. The CLS position is never masked.

---

## Config

All hyperparameters in a single `config.py`:

```python
# Architecture
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
FFN_DIM = 256
DROPOUT = 0.1
TIME_EMBED_L = 64           # sinusoidal frequencies for time encoding
T_DIFF = 1000               # diffusion timesteps
MAX_SEQ_LEN = 72            # 72-hour window
N_FEATURES = 39             # 34 time-varying + 5 static

# Training
BATCH_SIZE = 64
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PRETRAIN_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 10  # on validation AUPRC
FINETUNE_LR_RATIOS = [0.1, 0.3, 0.7, 1.0]   # per transformer layer, earliest first
                                               # 0.0 = frozen, 1.0 = full base LR

# ContiMask attribution
MASK_HIDDEN_DIM = 16        # MaskFunctionFourier hidden dim (matches paper)
MASK_L = 12                 # Fourier frequencies for mask (matches paper)
MASK_TARGET_AREA = 0.1      # cover 10% of observed data
MASK_EPOCHS = 200           # PGPE iterations (matches paper)
```

`FINETUNE_LR_RATIOS` has one entry per transformer layer. A ratio of `0.0` excludes that layer from the optimiser (frozen). The classification head always trains at full `FINETUNE_LR`.

---

## Training

### Phase flag

`--phase pretrain | finetune | both`

- `pretrain`: trains backbone + denoising head from random init, saves to `checkpoints/pretrained_backbone.pt`
- `finetune`: loads `checkpoints/pretrained_backbone.pt` if it exists, otherwise trains from random init; saves best checkpoint (by validation AUPRC) to `checkpoints/best_model.pt`
- `both`: runs pretrain then finetune sequentially

### Phase 1 — Pretraining

- All patients, regardless of label
- Sample `s ~ Uniform(0, T_DIFF)` per batch
- Linear DDPM noise schedule: corrupt **only observed features** (`data_mask == 1`) with Gaussian noise scaled by `sqrt(1 - ᾱ_s)`; NaN-filled and padded positions receive no noise
- Loss: MSE between predicted and clean features at observed positions only
- No class imbalance issue — labels are not used
- EarlyStopping on validation denoising loss (patience=10)

### Phase 2 — Finetuning

- Detach / discard the denoising head; attach classification head
- BCE loss with `pos_weight = n_neg / n_pos` to handle class imbalance
- AdamW with per-layer LR groups built from `FINETUNE_LR_RATIOS`; classification head at full `FINETUNE_LR`
- EarlyStopping on validation AUPRC (patience=10)
- Save `checkpoints/best_model.pt` and `checkpoints/norm_stats.pt`

---

## ContiMask Interface

All 39 features are in `X` and attributed over uniformly. No special static handling needed:

```python
def forward_func(t, X, data_mask):
    """
    Handles both standard (B, T, F) and stochastic K-sample (K, B, T, F) inputs
    from Deletion perturbation.
    """
    if X.dim() == 4:
        K, B, T, F = X.shape
        logit = model.classify(
            t.view(K * B, T),
            X.view(K * B, T, F),
            data_mask.view(K * B, T, F),
        )
        return torch.sigmoid(logit).view(K, B)
    return torch.sigmoid(model.classify(t, X, data_mask))  # (B,)
```

**Attribution config (matching ContiMask paper's sepsis experiments):**
- Mask: `MaskFunctionFourier(hidden_dim=16, L=12, features=39)`
- Perturbations: Deletion (PGPE), FadeMovingAverage (PGPE), GaussianBlur (PGPE)
- `target_area = 0.1`
- `n_epochs = 200`

---

## File Structure

```
contimask/
├── config.py                        # all hyperparameters
├── data/
│   ├── dataset.py                   # SepsisDataset: loads PSV, builds (t, X, data_mask, label)
│   └── split.py                     # patient-level stratified train/val/test split
├── models/
│   ├── diffusion_transformer.py     # DiffusionTransformer (backbone + heads)
│   └── embeddings.py                # SinusoidalTimeEmbedding, DiffusionStepEmbedding
├── training/
│   ├── pretrain.py                  # Phase 1 training loop
│   ├── finetune.py                  # Phase 2 training loop
│   └── train.py                     # CLI entry point with --phase flag
├── attribution/                     # existing ContiMask code (unchanged)
│   ├── mask_conti.py
│   └── perturbation_conti.py
├── sepsis_attribution.py            # forward_func + ContiMask runner for sepsis
├── checkpoints/                     # saved model weights and norm_stats.pt
├── results/                         # existing results directory
└── utils/
    ├── losses.py                    # existing
    ├── metrics.py                   # existing
    ├── tensor_manipulation.py       # existing
    └── tools.py                     # existing (EarlyStopping)
```

---

## Verification

1. **Data pipeline:** Load dataset, check `X.shape == (72, 39)`, `data_mask.shape == (72, 39)`, `data_mask[:, 34:].all() == True` for non-padded rows
2. **Model forward pass:** Instantiate `DiffusionTransformer`, run a batch of random inputs, check output shape `(B, 1)` with no NaN
3. **Pretraining:** `python training/train.py --phase pretrain --pretrain_epochs 2` — loss should decrease, checkpoint saved
4. **Finetuning:** `python training/train.py --phase finetune` — AUPRC reported on validation set, EarlyStopping triggers
5. **Attribution:** Run `sepsis_attribution.py` on one test patient, check `mask.get_mask(t)` shape is `(1, 72, 39)`
6. **ContiMask K>1 path:** Pass `X` with shape `(K, B, T, 39)` into `forward_func`, verify output shape `(K, B)`
