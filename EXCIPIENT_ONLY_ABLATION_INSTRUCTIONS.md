# Excipient-Only Ablation — Implementation Instructions

## 0. Context (read this first)

This repo (`ablation-exp-mlp-molformer-asl`) is a clone of the main API–Excipient
compatibility model. It must be converted into an **excipient-only ablation
model**: the exact same architecture, but the API branch is starved of all
real information (structure + descriptors) and replaced by a single learned
constant "API placeholder," mirroring the `exc_placeholder` /
`exc_global_placeholder` mechanism that already exists in `src/model.py` for
missing excipients.

Purpose: determine whether the main model's predictive power comes from
both API and excipient information, or from excipient information alone. A
second ablation (API-only, i.e. starving the excipient branch) will be done
later in a separate clone — **do not build that here**.

### Hard constraints — do not deviate

1. Do **not** remove or resize any layer: `api_proj`, `exc_proj`,
   `attn_exc_to_api`, `attn_api_to_exc`, `api_desc_proj`, `exc_desc_proj`,
   the classifier — all stay exactly as defined today (same shapes, same
   depth, same head count).
2. Do **not** add any new CLI flags, config toggles, or "ablation mode"
   switches. The starvation of the API branch must be **unconditional** —
   it applies to every single example, in every split (train/val/test),
   every time, with no way to turn it off. This is a fixed, separate repo
   dedicated only to this ablation.
3. Do **not** touch the excipient-side missing-data mechanism
   (`exc_placeholder`, `exc_global_placeholder`, `exc_available`,
   `modality_dropout_rate`). It is unrelated and must keep behaving exactly
   as it does today.
4. Only one source file requires a logic change: `src/model.py`. One
   convenience config change (checkpoint/metrics directory names) is made
   in `src/config.py`. No other file needs to change.
5. Do not touch `data/train.csv`, `data/val.csv`, `data/test.csv`,
   `data/api_descriptors.csv`, `data/excipient_descriptors.csv`, or
   `models/descriptor_norm_stats.json`. The ablation is implemented purely
   inside the model's forward pass, not in the data pipeline.

---

## 1. Clean the repo

This clone only needs to keep source code, configs, and raw data. Delete
everything that is a stale artifact from the main model's previous runs
(the task description says not to worry about preserving previous
checkpoints/metrics/outputs). Run exactly this from the
`ablation-exp-mlp-molformer-asl/` repo root:

```bash
# Stale training artifacts from the main model — this repo will produce its own
rm -rf checkpoints/molformer
rm -rf metrics/molformer
rm -rf outputs/*

# Cached bytecode / notebook checkpoint junk — safe to delete, regenerated automatically
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
rm -rf .hf_cache

# Exploratory notebook tied to the main model's narrative — not needed for this
# script-driven ablation workflow and will otherwise get confusing/stale
rm -f Note-book.ipynb

# Naive-split sanity-check artifacts (Section 9 leakage check) are orthogonal
# to this ablation; leave data/naive/ and sanity_check.py in place untouched,
# do not delete them, do not run them as part of this task.
```

Do **not** delete: `data/train.csv`, `data/val.csv`, `data/test.csv`,
`data/start_dataset.csv`, `data/api_descriptors.csv`,
`data/excipient_descriptors.csv`, `data/naive/`, `models/`, `.git/`,
`src/`, `main.py`, `cross_validate.py`, `sweep_asl.py`, `sanity_check.py`,
`smoke_test.py`, `scripts/`.

After cleanup, recreate empty target dirs (they'll be filled by training):

```bash
mkdir -p checkpoints metrics
```

---

## 2. Modify `src/model.py`

### 2.1 What changes and why

Today, `_prepend_cls` projects the API SMILES tokens + pooled embedding
through `api_proj` into `api_seq` (shape `[B, T+1, proj_dim]`), and that
tensor is used as-is in cross-attention and pooled into `h_api_struct`.

You will add **two new learned parameters** that mirror
`exc_placeholder` / `exc_global_placeholder`:

- `api_placeholder` — `nn.Parameter(shape [1, 1, proj_dim])`, stands in for
  the per-token API structural embedding.
- `api_global_placeholder` — `nn.Parameter(shape [1, proj_dim])`, stands in
  for the pooled/CLS API structural embedding.

And **one new learned parameter** for the descriptor branch:

- `api_desc_placeholder` — `nn.Parameter(shape [1, desc_proj_dim])`, stands
  in for the projected API descriptor vector (only created when
  `config.use_descriptors` is `True`, matching how `api_desc_proj` is only
  created in that case).

In `forward()`, `api_proj` and `api_desc_proj` still run on every batch
exactly as before (same FLOPs, same graph shape, same parameter count as
the main model, nothing skipped) — but their outputs are then
**unconditionally discarded and overwritten** with the placeholders,
broadcast to the batch. Cross-attention (`attn_exc_to_api`,
`attn_api_to_exc`) and the classifier run completely unmodified on the
resulting tensors — they just always see the same constant "API" vector
instead of real molecule-specific information. Because every single row
uses the placeholder (not just rows with a missing-data flag, since there
is no such flag for API), this is the excipient-side mechanism's `if row
is missing` branch made unconditional.

This adds 3 tiny parameter tensors (128 + 128 + 24 = 280 scalars with the
current config) on top of the existing model — the same kind of addition
`exc_placeholder`/`exc_global_placeholder` already made to the main model,
so it does not break "same parameter count" in the sense that matters:
no layer is resized, no head is removed, no depth changes.

### 2.2 Exact edit — `__init__`

In `src/model.py`, immediately after this existing block:

```python
        self.exc_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
        self.exc_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)
```

insert:

```python
        # --- Excipient-only ablation: learned API placeholder ---------- #
        # Mirrors exc_placeholder / exc_global_placeholder above, but for
        # the API branch, and is applied unconditionally to every example
        # (see forward()) instead of only to rows with missing data.
        self.api_placeholder = nn.Parameter(torch.randn(1, 1, proj_dim) * 0.01)
        self.api_global_placeholder = nn.Parameter(torch.randn(1, proj_dim) * 0.01)
```

Then, inside the existing `if config.use_descriptors:` block that creates
`self.api_desc_proj` / `self.exc_desc_proj`, add the descriptor placeholder
right after `self.exc_desc_proj` is created:

```python
        if config.use_descriptors:
            self.api_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )
            self.exc_desc_proj = _projection(
                config.num_descriptors, 32, config.desc_proj_dim, config.desc_dropout
            )
            self.api_desc_placeholder = nn.Parameter(
                torch.randn(1, config.desc_proj_dim) * 0.01
            )
```

### 2.3 Exact edit — `forward()`

Replace the current start of `forward()`:

```python
    def forward(self, batch):
        api_seq, api_mask = self._prepend_cls(
            batch["api_tokens"],
            batch["api_global"],
            batch["api_mask"],
            self.api_proj,
        )
        exc_seq, exc_mask = self._prepend_cls(
```

with:

```python
    def forward(self, batch):
        api_seq, api_mask = self._prepend_cls(
            batch["api_tokens"],
            batch["api_global"],
            batch["api_mask"],
            self.api_proj,
        )

        # ------------------------------------------------------------ #
        # Excipient-only ablation: unconditionally starve the API branch.
        # api_proj already ran above (identical compute graph to the main
        # model); its output is discarded here and replaced with fixed,
        # learned placeholder vectors for every example. Cross-attention
        # below still runs at full depth/width against this constant, so
        # it structurally cannot recover any real API information.
        # ------------------------------------------------------------ #
        api_batch_size = api_seq.size(0)
        api_seq = torch.zeros_like(api_seq)
        api_seq[:, :1, :] = self.api_global_placeholder.expand(api_batch_size, -1).unsqueeze(1)
        if api_seq.size(1) > 1:
            api_seq[:, 1:2, :] = self.api_placeholder.expand(api_batch_size, -1, -1)
            api_placeholder_mask = torch.ones_like(api_mask)
            api_placeholder_mask[:, 0] = False
            api_placeholder_mask[:, 1] = False
            api_mask = api_placeholder_mask

        exc_seq, exc_mask = self._prepend_cls(
```

Everything from `exc_seq, exc_mask = self._prepend_cls(...)` through the
two cross-attention calls stays **completely unchanged**.

Then find this block further down:

```python
        if self.config.use_descriptors:
            d_api = self.api_desc_proj(batch["api_desc"])
            d_exc = self.exc_desc_proj(batch["exc_desc"])
            h_api = torch.cat([h_api_struct, d_api], dim=-1)
            h_exc = torch.cat([h_exc_struct, d_exc, exc_avail], dim=-1)
```

and replace it with:

```python
        if self.config.use_descriptors:
            d_api = self.api_desc_proj(batch["api_desc"])  # computed, then discarded (ablation)
            d_api = self.api_desc_placeholder.expand(api_batch_size, -1)
            d_exc = self.exc_desc_proj(batch["exc_desc"])
            h_api = torch.cat([h_api_struct, d_api], dim=-1)
            h_exc = torch.cat([h_exc_struct, d_exc, exc_avail], dim=-1)
```

Nothing else in `forward()` changes: `refined_exc`/`exc_out`,
`refined_api`/`api_out`, `h_api_struct`, `h_exc_struct`, the
`interaction`/`difference` terms, `pair_vec`, and `self.classifier(pair_vec)`
all stay byte-for-byte identical to today.

### 2.4 Also add a one-line module docstring note

At the very top of `src/model.py`, change:

```python
"""MoLFormer-based API/excipient compatibility model."""
```

to:

```python
"""MoLFormer-based API/excipient compatibility model.

ABLATION VARIANT: the API branch is unconditionally replaced with a fixed
learned placeholder (see api_placeholder / api_global_placeholder /
api_desc_placeholder below). The model only ever sees excipient structure
and descriptors; it cannot use real API information. Architecture, layer
shapes, and depth are otherwise identical to the main model.
"""
```

---

## 3. Modify `src/config.py`

Keep this ablation run's checkpoints/metrics separate from anything else so
comparison against the main model's saved metrics is unambiguous. Change:

```python
    checkpoint_dir: str = "checkpoints/molformer"
    metrics_dir: str = "metrics/molformer"
```

to:

```python
    checkpoint_dir: str = "checkpoints/molformer_exc_only_ablation"
    metrics_dir: str = "metrics/molformer_exc_only_ablation"
```

Do not change any other field in `Config` (all hyperparameters — `lr`,
`batch_size`, `max_epochs`, ASL settings, dropout, `proj_dim`, `num_heads`,
etc. — must stay identical to the main model's config for a fair
comparison).

---

## 4. Verify the change before training

Run these two checks from the repo root, in order. Both must pass with no
errors before you start training.

### 4.1 Smoke test — forward pass shape/loss sanity

```bash
python smoke_test.py
```

Expect it to print a logits shape of `[4]` (batch size 4) and a finite loss
value, with no exceptions.

### 4.2 Placeholder-behavior check

Create and run this one-off script (delete it afterward, do not commit it
as a permanent repo file) to confirm two things: (a) the API branch is
truly constant regardless of input, and (b) the excipient branch is not
constant, i.e. the ablation only touches the API side.

```bash
cat > /tmp/verify_ablation.py << 'EOF'
import torch
from src.config import Config
from src.model import APIExcipientModel
from src.molformer_featurization import MolFormerFeaturizer

config = Config(use_descriptors=True)
device = torch.device("cpu")
featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path).to(device)
model = APIExcipientModel(config).to(device).eval()

def make_batch(api_smi, exc_smi):
    api_padded, api_global, api_mask, _ = featurizer([api_smi])
    exc_padded, exc_global, exc_mask, _ = featurizer([exc_smi])
    return {
        "api_tokens": api_padded, "api_global": api_global, "api_mask": api_mask,
        "exc_tokens": exc_padded, "exc_global": exc_global, "exc_mask": exc_mask,
        "exc_available": torch.tensor([1.0]),
        "api_desc": torch.randn(1, config.num_descriptors),
        "exc_desc": torch.randn(1, config.num_descriptors),
    }

with torch.no_grad():
    out_a = model(make_batch("CCO", "O"))
    out_b = model(make_batch("c1ccccc1C(=O)O", "O"))  # totally different API SMILES, same excipient
    out_c = model(make_batch("CCO", "CC(=O)O"))        # same API, different excipient

assert torch.allclose(out_a, out_b, atol=1e-6), (
    "FAIL: changing the API SMILES changed the output — API branch is not fully starved."
)
assert not torch.allclose(out_a, out_c, atol=1e-6), (
    "FAIL: changing the excipient did not change the output — excipient branch is broken."
)
print("PASS: output is invariant to API SMILES and sensitive to excipient SMILES, as expected.")
EOF
python /tmp/verify_ablation.py
rm /tmp/verify_ablation.py
```

If either assertion fails, stop and re-check the edits in Section 2 before
proceeding — do not train on a broken ablation.

### 4.3 Parameter count sanity

```bash
python - << 'EOF'
from src.config import Config
from src.model import APIExcipientModel
from src.utils import count_parameters

model = APIExcipientModel(Config())
trainable, total = count_parameters(model)
print(f"Trainable: {trainable:,} / Total: {total:,}")
EOF
```

Record this number. It should be the main model's original parameter count
plus exactly 280 (128 + 128 + 24, given `proj_dim=128`,
`desc_proj_dim=24`). Report this alongside your results so the parameter
counts of both models can be compared directly.

---

## 5. Train the ablation model

From the repo root:

```bash
python main.py
```

This uses `src/train.py`, `src/evaluate.py`, and `src/loss.py` completely
unmodified — no changes were needed there, since they only call
`model(batch)` and never touch API/excipient-specific internals. Training
will:

- Save the best checkpoint to `checkpoints/molformer_exc_only_ablation/best_model.pt`.
- Tune the decision threshold on the validation set.
- Evaluate on the held-out test set.
- Print and save a run summary to `metrics/molformer_exc_only_ablation/run_metrics.json`.

If you also want the 5-fold cross-validation numbers for the ablation
model (to compare against the main model's `cv_metrics.json`), run:

```bash
python cross_validate.py
```

This will pick up the updated `config.checkpoint_dir` /
`config.metrics_dir` automatically since it imports `Config` from
`src/config.py` — no changes needed to `cross_validate.py` itself.

Do not run `sanity_check.py` or `sweep_asl.py` as part of this task; they
are for unrelated purposes (API-identity leakage check, and ASL
hyperparameter sweeping) and are out of scope here.

---

## 6. Compare against the main model

You need the main model's `run_metrics.json` (and, if available,
`cv_metrics.json`) from the original (non-cloned) repo to compare against.
Ask the user for that file's contents/path if it is not already present
somewhere accessible to you — do not fabricate or estimate the main
model's numbers.

Once you have both, produce a short comparison covering:

- PR-AUC, F1, MCC, and any other metrics present in `run_metrics.json`,
  main model vs. excipient-only ablation, on both val and test splits.
- The parameter-count numbers from Section 4.3 for both models.
- One sentence of interpretation: if the ablation's metrics drop
  substantially, the main model relies meaningfully on API information; if
  they are close to the main model's, most of the signal is coming from
  the excipient alone.

Save this comparison as `ABLATION_RESULTS.md` in the repo root, and do not
delete or overwrite `metrics/molformer_exc_only_ablation/run_metrics.json` — it
is the primary evidence for the comparison.

---

## 7. Final checklist before declaring done

- [ ] `src/model.py` has the three new parameters and the two `forward()`
      edits from Section 2, and nothing else in that file changed.
- [ ] `src/config.py` has the two renamed directory fields from Section 3,
      and nothing else in that file changed.
- [ ] No file other than `src/model.py` and `src/config.py` was modified.
- [ ] Section 4.2's script passed (output invariant to API SMILES, varies
      with excipient SMILES).
- [ ] `python main.py` completed and produced
      `metrics/molformer_exc_only_ablation/run_metrics.json`.
- [ ] `ABLATION_RESULTS.md` written with the main-vs-ablation comparison.
