"""Mask-only VIQ18 reference: no interpolated or neural values are consumed."""
from __future__ import annotations
import numpy as np

def observed_viq18(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    x, m = np.asarray(values, dtype=np.float64), np.asarray(observed)
    if x.shape != (50, 128, 3) or m.shape != x.shape or m.dtype != bool:
        raise ValueError("H50 values and aligned boolean mask must be [50,128,3]")
    if not np.isfinite(x[m]).all() or not m.any(axis=1).all():
        raise ValueError("each cycle/channel needs at least one finite observed value")
    t = np.linspace(0., 1., 50)
    t -= t.mean()
    result = []
    for ch in range(3):
        # Boolean indexing excludes all hidden entries, including NaN sentinels.
        visible = x[:, :, ch][m[:, :, ch]]
        cycle_means = np.array([x[j, :, ch][m[j, :, ch]].mean() for j in range(50)])
        first = x[:5, :, ch][m[:5, :, ch]].mean()
        last = x[-5:, :, ch][m[-5:, :, ch]].mean()
        result.extend((visible.mean(), visible.std(ddof=0), visible.min(), visible.max(),
                       last-first, np.dot(t, cycle_means)/np.dot(t, t)))
    return np.asarray(result, dtype=np.float64)
