"""Backend fitting extracted from run_backend_grid_tuning_20261017.py."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge

def fit_backend(head, alpha, gamma=None):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("positive finite alpha required")
    if head["family"] == "ridge":
        model = Ridge(alpha=alpha, solver="svd").fit(head["x"], head["y"], sample_weight=head["weights"])
        return model, model.predict(head["query"])
    if gamma is None or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("positive finite RBF gamma required")
    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma).fit(
        head["x"], head["y"]-head["target_mean"], sample_weight=head["weights"])
    return model, model.predict(head["query"])+head["target_mean"]

