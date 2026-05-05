from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_dataset(data: Dict, device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        "train_input": torch.tensor(data["X_train"], dtype=torch.float32, device=device),
        "train_label": torch.tensor(data["y_train"], dtype=torch.float32, device=device),
        "test_input": torch.tensor(data["X_test"], dtype=torch.float32, device=device),
        "test_label": torch.tensor(data["y_test"], dtype=torch.float32, device=device),
    }


def predict_numpy(model, X, device: str = "cpu"):
    model_device = torch.device(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=model_device)
    with torch.no_grad():
        y = model(Xt)
    return y.detach().cpu().numpy()
