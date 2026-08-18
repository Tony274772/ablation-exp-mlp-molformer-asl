"""Run inference on the FULL data/val.csv and data/test.csv splits (not just
the small hand-picked outputs/validation_pairs.csv sample), and write:

  outputs/val_with_preds.csv    / outputs/test_with_preds.csv
      -> original columns + logit, prob, pred (threshold 0.5)

  outputs/val_api_embeddings.npy  / outputs/test_api_embeddings.npy
  outputs/val_exc_embeddings.npy  / outputs/test_exc_embeddings.npy
      -> (N, 768) float32 arrays of the frozen MoLFormer pooler embeddings
         for the API / excipient SMILES in each row, in the same row order
         as the corresponding *_with_preds.csv (join on row index, or on
         API_CID / Excipient_CID which are also in the csv).

Usage (from repo root, same as the original infer_validation_pairs.py):
    python scripts/infer_full_with_embeddings.py

Requires the real MoLFormer weights to be present at
models/molformer-xl-both-10pct/model.safetensors (run
`python models/download_molformer.py` or `git lfs pull` first if that file
is currently just a tiny LFS pointer).
"""
import os
import logging

import numpy as np
import pandas as pd
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runtime import configure_thread_limits, configure_torch_runtime
configure_thread_limits()
import torch as _torch
configure_torch_runtime(_torch)

from src.config import Config
from src.molformer_featurization import MolFormerFeaturizer
from src.dataset import get_dataloader_from_dataframe
from src.model import APIExcipientModel


def run_split(split_name, config, featurizer, model, device, root):
    inp_path = os.path.join(root, "data", f"{split_name}.csv")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    df = pd.read_csv(inp_path)
    logging.info(f"[{split_name}] loaded {len(df)} rows from {inp_path}")

    loader = get_dataloader_from_dataframe(config, featurizer, df, is_train=False, shuffle=False)

    all_logits = []
    api_emb_chunks = []
    exc_emb_chunks = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            logits = model(batch)
            all_logits.extend(logits.detach().cpu().numpy().tolist())

            # api_global / exc_global are the frozen MoLFormer pooler
            # embeddings (768-dim) computed inside the collate_fn's
            # featurizer call -- these are the "embeddings" for each
            # API / excipient SMILES in the batch.
            api_emb_chunks.append(batch["api_global"].detach().cpu().numpy())
            exc_emb_chunks.append(batch["exc_global"].detach().cpu().numpy())

    logits = np.array(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    out_df = df.copy()
    out_df["logit"] = logits
    out_df["prob"] = probs
    out_df["pred"] = preds

    out_csv = os.path.join(root, "outputs", f"{split_name}_with_preds.csv")
    out_df.to_csv(out_csv, index=False)

    api_arr = np.concatenate(api_emb_chunks, axis=0)
    exc_arr = np.concatenate(exc_emb_chunks, axis=0)

    api_emb_path = os.path.join(root, "outputs", f"{split_name}_api_embeddings.npy")
    exc_emb_path = os.path.join(root, "outputs", f"{split_name}_exc_embeddings.npy")
    np.save(api_emb_path, api_arr)
    np.save(exc_emb_path, exc_arr)

    logging.info(f"[{split_name}] wrote predictions -> {out_csv}")
    logging.info(f"[{split_name}] wrote API embeddings {api_arr.shape} -> {api_emb_path}")
    logging.info(f"[{split_name}] wrote excipient embeddings {exc_arr.shape} -> {exc_emb_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    # repo_root/scripts/infer_full_with_embeddings.py -> repo_root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = Config()
    device = config.get_device()
    logging.info(f"Device: {device}")

    featurizer = MolFormerFeaturizer(model_path=config.molformer_model_path).to(device)

    model = APIExcipientModel(config).to(device)
    ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        logging.info(f"Loaded checkpoint: {ckpt}")
    else:
        logging.warning(f"Checkpoint not found at {ckpt}; model remains randomly initialized.")
    model.eval()

    os.makedirs(os.path.join(root, "outputs"), exist_ok=True)

    for split in ["val", "test"]:
        run_split(split, config, featurizer, model, device, root)


if __name__ == "__main__":
    main()
