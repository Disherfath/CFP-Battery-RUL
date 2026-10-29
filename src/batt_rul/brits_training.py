"""Exact BRITS loss/update from the paper experiment runner."""
import numpy as np
import torch
ARMS = ("base","masked","uniform","task")
BASE_DIVISOR, AUX_WEIGHT = 2304., .1

def loss_terms(result, raw_clean, observed, channel_mean, channel_std, metric, arm):
    """Restore real observations before computing any differentiable features."""
    if arm not in ARMS or raw_clean.shape != (1, 50, 128, 3) or raw_clean.dtype != torch.float64:
        raise ValueError("one complete float64 H50 cell and a known arm required")
    if observed.shape != raw_clean.shape or observed.dtype != torch.bool or observed.all():
        raise ValueError("aligned observed mask with artificial hidden positions required")
    prediction = result["imputed_data"].reshape_as(raw_clean)
    target_z = ((raw_clean - channel_mean) / channel_std).float()
    completed = torch.where(observed, raw_clean, prediction.double() * channel_std + channel_mean)
    if not torch.isfinite(completed).all():
        raise FloatingPointError("nonfinite restored raw completion")
    normalized_base = result["loss"] / BASE_DIVISOR
    hidden_mse = (prediction - target_z).square()[~observed].mean()
    uniform = metric.loss(completed, raw_clean, kind="uniform")
    task = metric.loss(completed, raw_clean, kind="task")
    auxiliary = AUX_WEIGHT * (uniform if arm == "uniform" else task) if arm in ("uniform", "task") else normalized_base * 0.
    total = normalized_base + (hidden_mse if arm != "base" else 0.) + auxiliary
    terms = {"loss": total, "author_loss": result["loss"], "normalized_base": normalized_base,
        "hidden_mse": hidden_mse, "uniform_feature_loss": uniform, "life_projection_loss": task,
        "weighted_auxiliary": auxiliary, "observed_reconstruction": result["reconstruction_loss"],
        "forward_ort": result["forward_ort"], "backward_ort": result["backward_ort"],
        "weighted_consistency": result["consistency_loss"]}
    if any(not torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("nonfinite training loss or diagnostic")
    return terms, completed

def fit_step(model, optimizer, raw_clean, observed, mean, std, metric, arm, probe_auxiliary=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    truth_z = ((raw_clean - mean) / std).float()
    visible_z = torch.where(observed, truth_z, 0.)
    result = model(visible_z.reshape(50, 128, 3), observed.reshape(50, 128, 3), stage="train")
    terms, completed = loss_terms(result, raw_clean, observed, mean, std, metric, arm)
    active = [p for p in model.parameters() if p.requires_grad]
    auxiliary_norm = None
    if probe_auxiliary and arm in ("uniform", "task"):
        grads = torch.autograd.grad(terms["weighted_auxiliary"], active, retain_graph=True, allow_unused=True)
        squares = [g.detach().double().square().sum() for g in grads if g is not None]
        auxiliary_norm = float(torch.stack(squares).sum().sqrt()) if squares else 0.
        if not np.isfinite(auxiliary_norm) or auxiliary_norm <= 0:
            raise FloatingPointError("feature auxiliary loss has no finite nonzero model gradient")
    terms["loss"].backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
        raise FloatingPointError("missing/nonfinite effective BRITS gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in active):
        raise FloatingPointError("nonfinite BRITS parameter after update")
    row = {key: float(value.detach()) for key, value in terms.items()}
    row.update(gradient_norm=float(gradient_norm), auxiliary_gradient_norm=auxiliary_norm,
               observed_values_preserved=bool(torch.equal(completed[observed], raw_clean[observed])))
    return row

