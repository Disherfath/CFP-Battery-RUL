"""Exact Joint-SAITS loss and update from joint_saits_rul_baseline_20261018.py."""
import numpy as np
import torch
ETAS = (.01, .1, .3)

def loss_terms(result, raw_clean, observed, mean, std, head, lifetime, eta):
    if eta not in ETAS or raw_clean.dtype != torch.float64 or raw_clean.shape != (1, 50, 128, 3):
        raise ValueError("one raw64 H50 cell and predefined eta required")
    if observed.shape != raw_clean.shape or observed.dtype != torch.bool or observed.all():
        raise ValueError("canonical missing-view mask required")
    torch.testing.assert_close(result["loss"], result["reconstruction_loss"] + result["imputation_loss"], rtol=0, atol=0)
    prediction = result["imputed_data"].reshape_as(raw_clean)
    restored = torch.where(observed, raw_clean, prediction.double() * std + mean)
    predicted_target = head(restored)
    target = head.standardized_target(lifetime).detach()
    if target.shape != predicted_target.shape:
        raise ValueError("one aligned label per physical cell required")
    lifetime_mse = (predicted_target - target).square().mean()
    terms = {"loss": result["loss"] + eta * lifetime_mse, "author_loss": result["loss"],
        "ort": result["reconstruction_loss"], "mit": result["imputation_loss"],
        "standardized_lifetime_mse": lifetime_mse, "weighted_rul_loss": eta * lifetime_mse}
    if any(not torch.isfinite(v).all() for v in terms.values()):
        raise FloatingPointError("nonfinite joint objective")
    return terms, restored

def fit_step(model, head, optimizer, raw_clean, observed, mean, std, lifetime, eta, *, probe=False):
    model.train(); head.train()
    optimizer.zero_grad(set_to_none=True)
    truth_z = ((raw_clean - mean) / std).float().reshape(50, 128, 3)
    result = model(torch.where(observed.reshape_as(truth_z), truth_z, 0.),
                   observed.reshape_as(truth_z), truth_z, stage="train")
    terms, restored = loss_terms(result, raw_clean, observed, mean, std, head, lifetime, eta)
    active_model = [p for p in model.parameters() if p.requires_grad]
    active = active_model + list(head.parameters())
    lifetime_grad = None
    if probe:
        gradients = torch.autograd.grad(terms["weighted_rul_loss"], active_model, retain_graph=True, allow_unused=True)
        squares = [g.detach().double().square().sum() for g in gradients if g is not None]
        lifetime_grad = float(torch.stack(squares).sum().sqrt()) if squares else 0.
        if not np.isfinite(lifetime_grad) or lifetime_grad <= 0:
            raise FloatingPointError("lifetime supervision does not reach imputer")
    terms["loss"].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active):
        raise FloatingPointError("nonfinite joint gradient")
    norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in active):
        raise FloatingPointError("nonfinite joint parameter")
    return {**{k: float(v.detach()) for k, v in terms.items()}, "gradient_norm": float(norm),
        "rul_gradient_to_imputer_norm": lifetime_grad,
        "observed_values_preserved": bool(torch.equal(restored[observed], raw_clean[observed]))}

@torch.no_grad()
def predict_joint(head, bank, device):
    head.eval()
    result = []
    for cell in bank:
        raw = torch.as_tensor(cell, dtype=torch.float64, device=device)
        result.append(head.predict_loglife(raw).cpu().numpy())
    return np.stack(result)

