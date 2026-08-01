from __future__ import annotations

from pathlib import Path
import numpy as np


class NumpyMLP:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=False)
        self.input_dim = int(data["input_dim"])
        self.hidden_size = int(data["hidden_size"])
        self.w0 = data["linear0_weight"].astype(np.float32, copy=False)
        self.b0 = data["linear0_bias"].astype(np.float32, copy=False)
        self.ln_w = data["layernorm_weight"].astype(np.float32, copy=False)
        self.ln_b = data["layernorm_bias"].astype(np.float32, copy=False)
        self.w1 = data["linear1_weight"].astype(np.float32, copy=False)
        self.b1 = data["linear1_bias"].astype(np.float32, copy=False)
        self.w2 = data["linear2_weight"].astype(np.float32, copy=False)
        self.b2 = data["linear2_bias"].astype(np.float32, copy=False)

    def predict(self, x) -> float:
        z = np.asarray(x, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]
        z = z @ self.w0.T + self.b0
        mean = z.mean(axis=-1, keepdims=True)
        var = ((z - mean) ** 2).mean(axis=-1, keepdims=True)
        z = ((z - mean) / np.sqrt(var + 1e-5)) * self.ln_w + self.ln_b
        z = np.maximum(z, 0.0)
        z = np.maximum(z @ self.w1.T + self.b1, 0.0)
        z = z @ self.w2.T + self.b2
        return float(z.reshape(-1)[0])


def load_value_checkpoint(path: Path, device: str = "cpu") -> NumpyMLP:
    return NumpyMLP(path)


def load_action_checkpoint(path: Path, device: str = "cpu") -> NumpyMLP:
    return NumpyMLP(path)


def load_checkpoint(path: Path, device: str = "cpu") -> NumpyMLP:
    return load_value_checkpoint(path, device)
