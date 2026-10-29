"""Channel scaling and inference copied from the paper experiment helpers."""
from dataclasses import dataclass
import numpy as np
import torch
from .saits_task_adapter import restore_observed

@dataclass(frozen=True)
class ChannelStats:
    fit_ids: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, clean_fit, fit_ids):
        values = np.asarray(clean_fit, dtype=np.float64)
        if values.shape != (len(fit_ids), 50, 128, 3) or len(fit_ids) < 2 or len(set(fit_ids)) != len(fit_ids) or not np.isfinite(values).all():
            raise ValueError("finite clean fitting H50 curves for unique cells required")
        std = values.std(axis=(0, 1, 2), ddof=0)
        return cls(tuple(fit_ids), values.mean(axis=(0, 1, 2)), np.where(std < 1e-8, 1., std))

    def transform(self, values):
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def receipt(self):
        return {"fit_ids": list(self.fit_ids), "channel_mean": self.mean.tolist(), "channel_std": self.std.tolist()}

@torch.no_grad()
def complete_queries(model, linear_bank, masks, stats, device):
    """Input is completion/mask only: no artificial hold truth or lifetime argument."""
    values = np.asarray(linear_bank, dtype=np.float64)
    if values.ndim != 5 or values.shape[1:] != (10, 50, 128, 3) or masks.shape != values.shape or masks.dtype != bool:
        raise ValueError("aligned ten-view query bank and bool masks required")
    if not np.isfinite(values).all() or not masks[:, 0].all():
        raise ValueError("finite bank and original clean view required")
    model.eval()
    result = values.copy()
    mean, std = (torch.as_tensor(array, dtype=torch.float64, device=device) for array in (stats.mean, stats.std))
    for i in range(len(values)):
        for view in range(1, 10):
            visible = torch.as_tensor(values[i, view], dtype=torch.float64, device=device)
            observed = torch.as_tensor(masks[i, view], dtype=torch.bool, device=device)
            # Mask BEFORE standardization/casting, so even extreme hidden LI
            # placeholders cannot overflow or enter the author inputs.
            x = torch.as_tensor(stats.transform(np.where(masks[i, view], values[i, view], stats.mean)), device=device)
            output = model(x, observed, stage="test")["imputed_data"]
            if not torch.isfinite(output).all():
                raise FloatingPointError("nonfinite test-stage SAITS completion")
            result[i, view] = restore_observed(output, visible, observed, mean, std).cpu().numpy()
    np.testing.assert_array_equal(result[masks], values[masks])
    np.testing.assert_array_equal(result[:, 0], values[:, 0])
    return result

