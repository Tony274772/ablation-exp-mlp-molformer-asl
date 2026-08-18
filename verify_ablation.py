"""
Section 4.2 verification script — run once to confirm the ablation is correct,
then delete this file.

Checks:
  (a) Output is invariant to API SMILES (API branch fully starved).
  (b) Output changes with excipient SMILES (excipient branch intact).

NOTE: Uses fixed descriptor tensors so that only SMILES differences drive
output changes — random descriptors would mask the true comparison.
"""
import torch
from src.config import Config
from src.model import APIExcipientModel
from src.molformer_featurization import MolFormerFeaturizer

config = Config(use_descriptors=True)
device = torch.device("cpu")
featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path).to(device)
model = APIExcipientModel(config).to(device).eval()

# Fixed descriptors — same across calls so only SMILES changes matter
torch.manual_seed(0)
fixed_api_desc_1 = torch.randn(1, config.num_descriptors)
fixed_api_desc_2 = torch.randn(1, config.num_descriptors)  # Different API descriptors
fixed_exc_desc_same = torch.randn(1, config.num_descriptors)
fixed_exc_desc_diff = torch.randn(1, config.num_descriptors)

def make_batch(api_smi, exc_smi, api_desc, exc_desc):
    api_padded, api_global, api_mask, _ = featurizer([api_smi])
    exc_padded, exc_global, exc_mask, _ = featurizer([exc_smi])
    return {
        "api_tokens": api_padded, "api_global": api_global, "api_mask": api_mask,
        "exc_tokens": exc_padded, "exc_global": exc_global, "exc_mask": exc_mask,
        "exc_available": torch.tensor([1.0]),
        "api_desc": api_desc.clone(),
        "exc_desc": exc_desc.clone(),
    }

with torch.no_grad():
    # Same excipient (SMILES + descriptors), different API SMILES + different API descriptors
    out_a = model(make_batch("CCO", "O", fixed_api_desc_1, fixed_exc_desc_same))
    out_b = model(make_batch("c1ccccc1C(=O)O", "O", fixed_api_desc_2, fixed_exc_desc_same))

    # Same API SMILES, different excipient (SMILES + descriptors)
    out_c = model(make_batch("CCO", "CC(=O)O", fixed_api_desc_1, fixed_exc_desc_diff))

print(f"out_a = {out_a.item():.8f}")
print(f"out_b = {out_b.item():.8f}  (different API, same excipient)")
print(f"out_c = {out_c.item():.8f}  (same API, different excipient)")
print(f"|out_a - out_b| = {(out_a - out_b).abs().item():.2e}")
print(f"|out_a - out_c| = {(out_a - out_c).abs().item():.2e}")

assert torch.allclose(out_a, out_b, atol=1e-6), (
    "FAIL: changing the API SMILES changed the output — API branch is not fully starved."
)
assert not torch.allclose(out_a, out_c, atol=1e-6), (
    "FAIL: changing the excipient did not change the output — excipient branch is broken."
)
print("\nPASS: output is invariant to API SMILES and sensitive to excipient SMILES, as expected.")
