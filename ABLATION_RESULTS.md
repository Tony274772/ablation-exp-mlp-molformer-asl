# Excipient-Only Ablation — Results

## Model Comparison

This ablation starves the API branch of all real information (structure + descriptors)
while keeping the full architecture (cross-attention, classifier, layer shapes/depth)
identical to the main model. The API branch receives only a fixed learned placeholder
for every example. The excipient branch is completely untouched.

### Parameter Counts

| Model | Trainable Parameters | Total Parameters |
|-------|---------------------|-----------------|
| Main model | _TODO (run check_param_count.py on main repo)_ | _TODO_ |
| Excipient-only ablation | _TODO (run check_param_count.py)_ | _TODO_ |

> The ablation model has exactly **280 extra scalars** (128 for `api_placeholder`,
> 128 for `api_global_placeholder`, 24 for `api_desc_placeholder`) — no layers were
> resized, removed, or added.

### Test Set Metrics

| Metric | Main Model | Exc-Only Ablation | Δ |
|--------|-----------|-------------------|---|
| PR-AUC | 0.6190 | 0.6424 | +0.0234 |
| F1 | 0.5663 | 0.6202 | +0.0539 |
| MCC | 0.5072 | 0.5952 | +0.0880 |
| Precision | 0.5281 | 0.7692 | +0.2411 |
| Recall | 0.6104 | 0.5195 | −0.0909 |
| Accuracy | 0.8929 | 0.9271 | +0.0342 |

### Validation Set Metrics

| Metric | Main Model | Exc-Only Ablation | Δ |
|--------|-----------|-------------------|---|
| PR-AUC | 0.6669 | 0.5935 | −0.0734 |
| F1 | 0.6452 | 0.5839 | −0.0613 |
| MCC | 0.6093 | 0.5515 | −0.0578 |
| Precision | 0.6329 | 0.6557 | +0.0228 |
| Recall | 0.6579 | 0.5263 | −0.1316 |
| Accuracy | 0.9347 | 0.9323 | −0.0024 |

### Training Details

| Property | Main Model | Exc-Only Ablation |
|----------|-----------|-------------------|
| Best epoch | 3 | 23 |
| Threshold | 0.561 | 0.569 |
| Val loss | 0.0307 | 0.0416 |
| Test loss | 0.0408 | 0.0475 |

## Interpretation

The ablation model performs **comparably to the main model**, with validation metrics
slightly lower (PR-AUC −7%) but test metrics comparable or even slightly higher
(PR-AUC +2%, F1 +5%, MCC +9%). The ablation model trades recall for substantially
higher precision on the test set.

This indicates that **excipient information alone carries most of the predictive
signal** for API–excipient compatibility in this dataset. The API branch in the
main model contributes marginal additional value. Certain excipients are broadly
problematic or broadly safe, and their structure/descriptors alone are highly
predictive of compatibility outcomes.

The fact that the ablation needed 23 epochs vs. 3 for the main model suggests the
model takes longer to converge without API information, but ultimately reaches
comparable performance using only excipient features.

## Verification Checks

- [x] Section 4.1 smoke test (`python smoke_test.py`) — passed
- [ ] Section 4.2 placeholder-behavior check (`python verify_ablation.py`) — pending rerun with fixed script
- [ ] Section 4.3 parameter count sanity — pending (`python check_param_count.py`)
- [x] Training completed (`python main.py`)
- [x] `metrics/molformer_exc_only_ablation/run_metrics.json` produced
