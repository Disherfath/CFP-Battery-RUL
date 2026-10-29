"""Exact T1 masked reconstruction loss/update from the paper runners."""
import numpy as np
import torch

def hidden_mse(prediction, target, observed):
    if (prediction.ndim != 4 or prediction.shape[1:] != (50, 128, 3) or not len(prediction)
            or target.shape != prediction.shape or observed.shape != prediction.shape
            or observed.dtype != torch.bool or not prediction.is_floating_point()
            or target.dtype != prediction.dtype or target.device != prediction.device or observed.device != prediction.device):
        raise ValueError("aligned same-device floating [B,50,128,3] and bool mask required")
    hidden = ~observed
    counts = hidden.flatten(1).sum(1)
    if (counts == 0).any() or not torch.isfinite(prediction).all() or not torch.isfinite(target[hidden]).all():
        raise FloatingPointError("every cell requires finite predictions and hidden targets")
    # where before subtraction excludes even nonfinite unused observed targets.
    error = torch.where(hidden, prediction, 0.) - torch.where(hidden, target, 0.)
    loss = (error.square().flatten(1).sum(1) / counts).mean()
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite complete-cell MIT MSE")
    return loss

def fit_step(model, optimizer, raw_clean, observed, mean, std):
    if raw_clean.dtype != torch.float64 or raw_clean.shape != (1, 50, 128, 3):
        raise ValueError("one complete float64 H50 cell required")
    if observed.dtype != torch.bool or observed.shape != raw_clean.shape or observed.all():
        raise ValueError("same-shape mask with artificial missing values required")
    model.train(); optimizer.zero_grad(set_to_none=True)
    visible = torch.where(observed, raw_clean, torch.zeros_like(raw_clean))
    target = ((raw_clean - mean) / std).float()
    predicted = model(visible, observed)
    loss = hidden_mse(predicted, target, observed)
    loss.backward()
    active = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
        raise FloatingPointError("missing/nonfinite native T1 gradient")
    norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in active):
        raise FloatingPointError("nonfinite native T1 parameter after update")
    return {"loss": float(loss.detach()), "gradient_norm": float(norm)}

