"""
Section 4.2 verification script — run once to confirm the ablation is correct,
then delete this file.

Checks:
  (a) Output is invariant to API SMILES (API branch fully starved).
  (b) Output changes with excipient SMILES (excipient branch intact).
"""
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
