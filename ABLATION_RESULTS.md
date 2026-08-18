# Excipient-Only Ablation — Results

## Model Comparison

This ablation starves the API branch of all real information (structure + descriptors)
while keeping the full architecture (cross-attention, classifier, layer shapes/depth)
identical to the main model. The API branch receives only a fixed learned placeholder
for every example.

### Parameter Counts

| Model | Trainable Parameters | Total Parameters |
|-------|---------------------|-----------------|
| Main model | _TODO_ | _TODO_ |
| Excipient-only ablation | _TODO_ | _TODO_ |

> The ablation model has exactly **280 extra scalars** (128 for `api_placeholder`,
> 128 for `api_global_placeholder`, 24 for `api_desc_placeholder`) — no layers were
> resized, removed, or added.

### Test Set Metrics

| Metric | Main Model | Exc-Only Ablation | Δ |
|--------|-----------|-------------------|---|
| PR-AUC | _TODO_ | _TODO_ | _TODO_ |
| F1 | _TODO_ | _TODO_ | _TODO_ |
| MCC | _TODO_ | _TODO_ | _TODO_ |
| Precision | _TODO_ | _TODO_ | _TODO_ |
| Recall | _TODO_ | _TODO_ | _TODO_ |
| ROC-AUC | _TODO_ | _TODO_ | _TODO_ |
| Threshold | _TODO_ | _TODO_ | — |

### Validation Set Metrics

| Metric | Main Model | Exc-Only Ablation | Δ |
|--------|-----------|-------------------|---|
| PR-AUC | _TODO_ | _TODO_ | _TODO_ |
| F1 | _TODO_ | _TODO_ | _TODO_ |
| MCC | _TODO_ | _TODO_ | _TODO_ |

## Interpretation

_TODO: If the ablation's metrics drop substantially, the main model relies
meaningfully on API information. If they are close to the main model's,
most of the signal is coming from the excipient alone._

## Verification Checks

- [ ] Section 4.1 smoke test (`python smoke_test.py`) — passed
- [ ] Section 4.2 placeholder-behavior check (`python verify_ablation.py`) — passed
- [ ] Section 4.3 parameter count sanity — ablation = main + 280
- [ ] Training completed (`python main.py`)
- [ ] `metrics/molformer_exc_only_ablation/run_metrics.json` produced
