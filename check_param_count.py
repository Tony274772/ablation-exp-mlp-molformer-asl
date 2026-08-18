"""
Section 4.3 — Parameter count sanity check.
Expected: main model params + 280 (128 + 128 + 24 from the 3 API placeholders).
Delete this file after confirming.
"""
from src.config import Config
from src.model import APIExcipientModel
from src.utils import count_parameters

model = APIExcipientModel(Config())
trainable, total = count_parameters(model)
print(f"Trainable: {trainable:,} / Total: {total:,}")
