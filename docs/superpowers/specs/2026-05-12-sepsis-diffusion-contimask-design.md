# Sepsis Diffusion Model + ContiMask Attribution — Design Document

**Date:** 2026-05-12
**Status:** Draft — approach not yet finalised; to be discussed with supervisor

---

## Goal

Train a model on the PhysioNet 2019 sepsis prediction dataset and use the ContiMask attribution framework to explain its predictions — identifying which time steps and clinical features drive the model's output for individual patients.

---

## Dataset

**Source:** PhysioNet Challenge 2019 (`training_setA` only, locally at `~/msc_ai/individual-project/sepsis_data/training_setA/`)

**Format:** One pipe-delimited `.psv` file per patient. Each row is one ICU hour.

**Columns:**
- 8 vital signs (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2)
- 26 lab values (blood gas, liver/kidney function, haematology, metabolic)
- 5 demographics/admin (Age, Gender, Unit1, Unit2, HospAdmTime)
- `ICULOS` — ICU length of stay in hours (the time axis)
- `SepsisLabel` — 1 for all hours ≥ t_sepsis − 6 (positive 6 hours before clinical diagnosis), else 0

**Key characteristics:**
- ~90% of feature values are NaN (irregular, sparse observations)
- Severe class imbalance: most patients never develop sepsis
- Variable-length sequences (different ICU stays)
- Three hospital systems in the full dataset (only one set available locally)

**Data representation for the model:**
- `t` — ICULOS normalised to [0, 1] per patient; used as the continuous time axis
- `X` — the 39 clinical features with NaN replaced by 0
- `data_mask` — binary tensor (1 = observed, 0 = missing or padded)
- **Label** — patient-level binary: 1 if `SepsisLabel` is ever 1 in the record

The patient-level label (rather than per-timestep) is used because ContiMask's `forward_func` must return a scalar per patient.

---

## ContiMask Interface Constraint

ContiMask requires a `forward_func(t, X, data_mask) -> tensor` where:
- `t`: `(B, T)` normalised time
- `X`: `(B, T, F)` features
- `data_mask`: `(B, T, F)` binary
- Returns: `(B,)` scalar prediction per patient

When using stochastic perturbations with K > 1 samples (e.g. Deletion), ContiMask calls `forward_func` with `(K, B, T)` shaped tensors. The wrapper must flatten to `(K*B, T)`, run the model, and reshape back to `(K, B)`.

This constraint rules out any model whose inference requires iterative sampling (e.g. full DDPM reverse chain), as ContiMask calls `forward_func` thousands of times per patient during optimisation.

---

## Diffusion Approaches Considered

### Option 1 — DDPM as representation learner (most strongly considered)

Train a transformer backbone with a DDPM denoising objective (self-supervised pretraining), then attach a classification head and fine-tune with weighted BCE.

**What "diffusion" means here:**
- During pretraining: corrupt observed features with Gaussian noise at a random diffusion step `s`, train the backbone to denoise
- During fine-tuning and inference: the denoising head is unused; `forward_func` calls only the classification head

**Pros:**
- Fastest inference — single transformer forward pass, no diffusion steps at query time
- Self-supervised pretraining uses all patients equally regardless of label, avoiding class imbalance during the feature-learning phase
- Clean ContiMask integration — `forward_func` is just `model.classify()`
- Well-understood training procedure with established convergence behaviour

**Cons:**
- Diffusion is only in the training objective — the "diffusion model" framing is misleading at inference time
- Two-phase training is slower overall than a single training run
- Pretraining may not meaningfully improve performance with ~20k labelled patients
- The denoising head is discarded after pretraining (wasted capacity)

---

### Option 2 — Generative classifier (class-conditional DDPM)

Train two separate DDPM models — one on sepsis-positive patients, one on sepsis-negative. Estimate P(sepsis | X) via Bayes' rule using approximate DDPM likelihoods.

**Pros:**
- Purely generative — no discriminative head needed, theoretically principled
- Diffusion is genuinely present at inference
- Attribution reflects which parts of the data shift the class-conditional likelihood, not just a learned decision boundary

**Cons:**
- DDPM likelihood estimation requires importance sampling or ELBO over many diffusion steps — expensive and approximate
- Prohibitively slow for ContiMask which calls `forward_func` thousands of times per patient
- Requires training two separate DDPM models (one per class)
- Approximate likelihoods introduce noise into the attribution signal

**Concern:** Inference speed is incompatible with the ContiMask query budget for a prototype.

---

### Option 3 — CSDI-style imputation → separate classifier

Use a conditional score-based diffusion model to impute missing values, then train a standard classifier on the imputed (complete) time series. ContiMask runs on the classifier.

**Two sub-variants:**

- **Wrap only the classifier (3A):** Pre-impute once, run ContiMask on the dense imputed series.
- **Wrap the full pipeline (3B):** Each ContiMask perturbation triggers a full CSDI re-imputation then a classifier forward pass.

**Pros:**
- Imputation is the most natural application of diffusion to irregular time series
- Clear separation of concerns — the classifier can be any architecture
- 3B gives attribution on original sparse observations, not imputed ones

**Cons (3A):**
- Attribution is on imputed values, not the original observations
- `data_mask` is all-ones after imputation — the missingness structure, and the information it carries, is lost
- ContiMask perturbations (e.g. Deletion) lose their meaning when applied to an already-dense series

**Cons (3B):**
- Each ContiMask query requires running full CSDI reverse diffusion (hundreds of denoising steps), multiplied by thousands of optimisation iterations — prohibitively slow
- CSDI sampling is stochastic, making the optimisation landscape noisy for gradient-based ContiMask
- Two separate models to train and maintain

**Concern:** 3A loses attribution quality on the key irregular-time-series dimension; 3B is likely too slow for a prototype.

---

### Option 4 — Anomaly detection (unsupervised)

Train a DDPM only on sepsis-negative patients, treating sepsis as an anomaly. The training objective is identical for both sub-variants; they differ only at inference.

**Training (both sub-variants):**
- DDPM denoising on sepsis-negative patients only
- Corrupt observed features at random diffusion step s ~ Uniform(0, 1000)
- Loss: MSE between predicted and clean features at observed positions only
- No class labels used; no classification head

**Pros (both sub-variants):**
- No labels needed during training — the diffusion model is trained purely generatively
- Attribution answers "which parts of this trajectory look abnormal relative to normal ICU patients?" — clinically interpretable
- The diffusion model is genuinely load-bearing at inference

**Shared concern:** Sepsis patients are not necessarily statistical anomalies — many non-sepsis ICU patients have equally alarming trajectories. If discrimination is poor, the attribution maps will not be meaningful regardless of sub-variant.

#### Option 4A — Reconstruction error

Use the expected reconstruction error under the reverse diffusion chain as the anomaly score: high error = unusual trajectory = likely sepsis.

**Pros:**
- Principled — directly measures how well the model can reconstruct the input as a normal ICU trajectory
- Clear interpretation of the anomaly signal

**Cons:**
- Requires running the full DDPM reverse chain (hundreds of steps) at inference
- Incompatible with the ContiMask query budget — too slow for thousands of `forward_func` calls per patient
- Reconstruction error is a noisy proxy for sepsis risk, with no direct optimisation toward the clinical objective

#### Option 4B — Score-function inference

Replace reconstruction error with noise prediction magnitude at a small fixed noise level as the anomaly score.

**Background — what the score function is:**
The score function of a probability distribution p(x) is ∇_x log p(x) — the gradient of the log-likelihood with respect to the data. In a trained DDPM, the noise predictor ε_θ(x_s, s) is proportional to the negative score:

```
ε_θ(x_s, s)  ≈  −√(1 − ᾱ_s) · ∇_{x_s} log p(x_s)
```

At s=small (near-clean data), this measures how well the input fits the learned distribution of normal ICU trajectories. Large predicted noise = the model considers the trajectory unusual = potentially septic.

**Inference:**
```
anomaly_score(X) = mean(|ε_θ(X, s=small)|)
```

This is a single forward pass — no reverse diffusion chain required, making it compatible with the ContiMask query budget.

**Pros:**
- Single forward pass — ContiMask-compatible
- Diffusion is genuinely present and load-bearing at inference; the denoising network is the classifier
- Avoids the reverse chain while retaining the generative model's distributional knowledge

**Cons:**
- Two proxy steps: noise magnitude approximates the score, which is itself a proxy for anomaly-ness
- The choice of s is a hyperparameter with no obvious principled value — too large and the model cannot distinguish normal from abnormal; too small and sensitivity may be insufficient
- Does not resolve the fundamental discrimination concern: sepsis patients may still not be statistical anomalies relative to the normal ICU population

---

### Option 5 — Joint training (single-phase, multi-task)

Same architecture as Option 1, but train the denoising and classification objectives simultaneously with a combined loss:

```
loss = λ · denoising_loss + (1 − λ) · classification_loss
```

**Pros:**
- Simpler pipeline — one training run rather than two sequential phases
- Classification signal is available from the start, potentially leading to faster convergence toward the end task
- No decision required about when to switch phases

**Cons:**
- The two objectives can interfere — class imbalance may cause the classification gradient to dominate early training, degrading the denoising signal before the backbone has learned useful representations
- Requires tuning an additional hyperparameter λ to balance the two losses
- Harder to diagnose: if performance is poor, it is unclear whether the denoising or the classification objective is the cause

**Less preferred:** Two-phase training keeps the objectives clean and is easier to debug. Joint training could be revisited if the two-phase approach proves too slow.

---

### Preliminary direction

Option 1 is the most strongly considered approach, pending discussion with a supervisor. The overriding constraint is that ContiMask calls `forward_func` thousands of times per patient: Options 2 and 3B face significant inference speed challenges; Option 3A weakens the attribution quality on the irregular-time-series dimension that is central to the project; Option 4A is incompatible with the ContiMask query budget; Option 4 carries meaningful risk of poor discrimination regardless of sub-variant; Option 5 introduces objective interference and an extra hyperparameter with no clear benefit over two-phase training for a first prototype. Option 1 gives the fastest forward pass, the cleanest ContiMask integration, and a well-understood training procedure.

The key limitation — that diffusion is confined to the pretraining objective and the model is a plain transformer classifier at inference — is an open question worth raising with a supervisor. A secondary direction also worth discussing: Option 4B addresses the inference speed problem of Option 4A, making the diffusion genuinely load-bearing at inference rather than only at training time. Option 4B is the strongest candidate if having real diffusion at inference is a requirement.

---

## Architecture

The architecture differs meaningfully between the options under consideration.

### Options 1 and 5 — DiffusionTransformer

A transformer encoder with:

**Input:** `(t, X, data_mask)`
- `t` encoded as sinusoidal embeddings at multiple frequencies (not integer positions — preserves actual time gaps between irregular observations)
- Each timestep token = `linear_projection(X_i) + sinusoidal(t_i)`

**CLS token:** A learnable parameter prepended to the sequence. After the transformer runs, the CLS output has attended to all other positions and provides a fixed-size sequence-level representation. Used by the classification head. (Concept from BERT.)

**Attention masking:** Positions where `data_mask.any(dim=-1) == False` (no feature observed, i.e. padded) are excluded from attention via `src_key_padding_mask`. Positions with partial observations are included.

**Diffusion step embedding:** The noise level `s` is sinusoidally encoded → MLP → added to every token. Tells the model how noisy the input is.
- *Options 1 and 5:* used during pretraining; omitted at inference (the denoiser head is discarded and only the classification head runs)
- *Option 5:* same as Option 1, just trained jointly rather than in two phases

**Denoising head:** Linear layer on sequence tokens `(B, T, d_model) → (B, T, 39)`. Predicts clean features from noisy input. Training-time only for Options 1 and 5 — not used at inference.

**Classification head:** Small MLP on CLS token `(B, d_model) → (B, 1)`. Used at fine-tuning and inference.

---

### Option 4B — Score-based anomaly detector

A different use of the same denoising network, without a classification head. The model is trained identically to Option 4A (DDPM on sepsis-negative patients only); only inference differs.

**Inference:** Evaluate the noise predictor on a clean patient time series at a small fixed noise level s=small:

```
anomaly_score(X) = mean(|ε_θ(X, s=small)|)
```

Large predicted noise = the model considers the trajectory unusual relative to normal ICU patients = potentially septic. This is a single forward pass — no CLS token or classification head needed, and no reverse diffusion chain.

**Denoising head:** Used at both training and inference — it is the prediction signal. This is the key difference from Options 1 and 5 where it is discarded after pretraining.

**Step embedding:** Used at both training and inference (passed with the fixed small s value).

---

### Shared hyperparameters (most strongly considered values)

| Parameter | Value |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 |
| `n_layers` | 4 |
| `ffn_dim` | 256 |
| `T_diff` (diffusion steps) | 1000 |
| Max sequence length | 336 hours (14 days) |

`d_model=64` preferred for prototype speed (vs 128 which would roughly double training time and memory).

---

## Architecture Alternatives Considered (Options 1 and 5)

The following alternatives apply to Options 1 and 5 only, where the backbone's job is to produce a CLS-style representation for classification. They do not apply to Option 4B, where the denoising network output is the prediction signal directly.

All three satisfy the ContiMask constraint (accept `(t, X, data_mask)`, return a scalar per patient).

**Transformer (most strongly considered):** Multi-head self-attention with CLS token aggregation and attention masking for padded positions. Natural fit for irregular time series via sinusoidal time embeddings. Higher parameter count and training cost than alternatives.

**GRU with sinusoidal time embedding:** Concatenate `sinusoidal(t_i)` to each feature vector before feeding into a GRU. Final hidden state → linear → sepsis logit. Fewer parameters, faster to train, well-established for clinical time series. Loses explicit attention, which is only relevant if attention maps are needed as a secondary interpretability signal (they aren't, since ContiMask provides attribution). A pragmatic choice if training speed is the primary concern.

**MLP per timestep + pooling:** Project each `(t_i, X_i)` independently through an MLP, mean-pool over observed timesteps, then classify. Simplest and fastest option. Less preferred because it discards temporal ordering — the trajectory of a patient's vitals over time carries clinical information that a permutation-invariant model cannot capture.

---

## Training

Training differs significantly between the two candidate approaches.

### Options 1 and 5

**Phase 1 — Pretraining (optional for prototype):**
- Uses all patients regardless of label
- Corrupt observed features at random diffusion step s ~ Uniform(0, 1000)
- Loss: MSE between predicted and clean features at observed positions only
- No class imbalance issue during this phase
- Can be skipped: start directly from Phase 2 with randomly initialised weights

**Phase 2 — Fine-tuning (or joint training for Option 5):**
- BCE with `pos_weight = n_neg / n_pos` to handle class imbalance
- Backbone LR = 0.1× classification head LR
- EarlyStopping (patience=10) reused from `utils/tools.py`
- Norm stats saved to `checkpoints/norm_stats.pt` for reuse at attribution time

**Why pretraining may not matter much:** With ~20k training patients and a direct classification objective + pos_weight, there is enough labelled data for direct supervised training. Pretraining is most valuable when labelled data is scarce.

### Option 4A and 4B

**Single training phase — DDPM on negatives only (identical for both sub-variants):**
- Uses only sepsis-negative patients
- Corrupt observed features at random diffusion step s ~ Uniform(0, 1000)
- Loss: MSE between predicted and clean features at observed positions only
- No class labels used during training
- No classification head — the denoising network is the entire model
- EarlyStopping on validation denoising loss (using held-out negative patients)

---

## Attribution

The ContiMask framework is the same regardless of which option is chosen, but the `forward_func` wrapper differs.

**Options 1 and 5:**
```python
def forward_func(t, X, data_mask):
    return torch.sigmoid(model.classify(t, X, data_mask))  # (B,)
```

**Option 4B:**
```python
S_FIXED = 10  # small fixed noise level

def forward_func(t, X, data_mask):
    s = torch.full((X.shape[0],), S_FIXED, dtype=torch.long)
    noise_pred = model.denoise(t, X, data_mask, s)          # (B, T, F)
    # Mean absolute predicted noise over observed positions as anomaly score
    score = (noise_pred.abs() * data_mask).sum(dim=(1, 2)) / data_mask.sum(dim=(1, 2)).clamp(min=1)
    return score                                             # (B,)
```

In both cases, the K > 1 path (used by Deletion perturbation) requires flattening `(K, B, T)` → `(K*B, T)` before calling the model and reshaping back to `(K, B)` afterwards.

The same explainer combinations from the existing synthetic experiments apply:

- **Masks:** MFF (Fourier), MFMLP, MT (tensor)
- **Perturbations:** D (Deletion), GB (Gaussian blur), FMA (Fade to moving average)
- **Optimisers:** G (gradient/Adam), E (EvoTorch/PGPE)

Deletion + EvoTorch (`MFF-D-E`) is the paper's key contribution for irregular time series — it simulates that observations were never collected, capturing informative missingness.

**Attribution bottleneck:** ContiMask's optimisation loop (2000–8000 steps per patient) is the dominant runtime cost regardless of which option or model architecture is used.

---

## Challenges of Applying ContiMask to a Diffusion Model

### 1. No natural scalar output

ContiMask requires `forward_func → (B,)` — a single scalar prediction per patient. Classifiers produce this naturally via a sigmoid. Diffusion models are generative: their outputs are samples, noise predictions (a vector of the same shape as the input), reconstruction errors, or likelihood bounds. Any of these require a design choice to collapse into a scalar, and that choice determines what the attribution actually measures.

### 2. The attribution target is a proxy

Whatever scalar is chosen, the attribution answers "which features change *this scalar* when perturbed?" — not necessarily "which features drive sepsis risk?". The two are related but not the same:

- **Reconstruction error:** "which features, when removed, change how well the model reconstructs the input?" Circular, because the removed features are part of what is being reconstructed.
- **Noise prediction magnitude (Option 4B):** "which features does the model predict would be noisier than expected for a normal ICU patient?" More meaningful for anomaly detection, but still a proxy for clinical risk.
- **Classifier logit (Options 1 and 5):** Directly interpretable as "which features drive the classification decision?" — but then the attribution is of the classifier, not of the diffusion model.

### 3. Diffusion is often only at training time

For Options 1 and 5, the diffusion is confined to the pretraining objective. At inference the model is a plain transformer classifier; ContiMask attributes the classifier's outputs. The framing as "attributing a diffusion model" is misleading — the attribution answers a discriminative question about a discriminative model.

Option 4B avoids this: the noise predictor is the prediction signal, so the diffusion model is genuinely present and load-bearing at inference.

### 4. Stochasticity

The diffusion forward process (`q_sample`) adds random noise. If this stochasticity enters `forward_func`, the optimisation landscape becomes noisy for gradient-based ContiMask. This is manageable by fixing the noise level `s` and not re-sampling noise inside `forward_func`, but requires care. EvoTorch (PGPE) is more tolerant of noisy objectives than gradient descent.

### 5. Perturbation validity under heavy missingness

~90% of features are already missing. ContiMask's Deletion perturbation removes further observed features, potentially pushing the input far off-distribution relative to what the model was trained on. With so few observed values to begin with, removing even a small number may leave the model with no signal, making the attribution gradient uninformative.

### 6. Option 4A/4B discrimination risk amplifies all of the above

If the anomaly score does not meaningfully separate septic from non-septic patients, ContiMask will produce attribution maps for a signal that is not predictive. The attribution machinery will run correctly, but the result will explain noise rather than clinical risk. This is the key empirical unknown for both Option 4 sub-variants.

---

## Open Questions

1. **Is the "diffusion model" framing justified for Options 1 and 5?** At inference these are plain transformer classifiers. The diffusion is only in the training objective (pretraining), and the denoiser head is unused at inference. Option 4B avoids this problem — the diffusion is genuinely load-bearing at inference. This framing question is worth raising with a supervisor.

2. **Option 4A/4B discrimination risk.** The anomaly detector may not discriminate well enough between septic and non-septic patients, since both groups can have alarming ICU trajectories. This is the key empirical unknown for both sub-variants.

3. **Skip pretraining in Options 1 and 5?** For a prototype, direct supervised training from random init is simpler and possibly equally effective with ~20k patients and pos_weight-corrected BCE.

4. **Joint training instead of two phases (Option 5)?** Simpler pipeline, but risks objective interference. Two-phase is safer.

5. **GRU vs transformer (Options 1 and 5)?** A GRU with sinusoidal time embedding is a drop-in replacement with no loss in ContiMask compatibility and significantly fewer parameters. Not applicable to Options 4A or 4B.

6. **Only `training_setA` is available.** The full dataset has 40,336 patients; only ~20,336 are local. This limits generalisability.
