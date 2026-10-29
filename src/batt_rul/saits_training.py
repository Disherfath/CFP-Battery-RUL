"""Original matched loss/update functions; historical module indirection removed."""
import numpy as np
import torch
ARMS = ("base", "uniform", "task")
AUX_WEIGHT = .1

def _base_loss_terms(result, raw_clean, observed, mean, std, metric, arm):
    if arm not in ARMS or raw_clean.shape != (1, 50, 128, 3) or raw_clean.dtype != torch.float64:
        raise ValueError("one complete float64 H50 physical cell and known arm required")
    if observed.dtype != torch.bool or observed.shape != raw_clean.shape or observed.all():
        raise ValueError("aligned observation mask with hidden entries required")
    # Exact original adapter objective: author ORT + MIT, without BRITS scaling.
    torch.testing.assert_close(result["loss"], result["reconstruction_loss"] + result["imputation_loss"], rtol=0, atol=0)
    prediction = result["imputed_data"].reshape_as(raw_clean)
    restored = torch.where(observed, raw_clean, prediction.double() * std + mean)
    uniform = metric.loss(restored, raw_clean, kind="uniform")
    direction = metric.loss(restored, raw_clean, kind="task")
    auxiliary = AUX_WEIGHT * (uniform if arm == "uniform" else direction) if arm != "base" else result["loss"] * 0.
    terms = {"loss": result["loss"] + auxiliary, "author_loss": result["loss"],
        "ort": result["reconstruction_loss"], "mit": result["imputation_loss"],
        "uniform_feature_loss": uniform, "life_projection_loss": direction,
        "weighted_auxiliary": auxiliary}
    if any(not torch.isfinite(value).all() for value in terms.values()):
        raise FloatingPointError("nonfinite author or auxiliary loss")
    return terms, restored

def loss_terms(result, raw_clean, observed, mean, std, metric, arm, auxiliary_weight=.1):
    # Reuse the already tested original loss/gradient implementation verbatim;
    # only the multiplier is explicit for the separate sensitivity experiment.
    terms, restored = _base_loss_terms(result, raw_clean, observed, mean, std, metric, arm)
    if arm != "base":
        auxiliary = auxiliary_weight * terms["uniform_feature_loss" if arm == "uniform" else "life_projection_loss"]
        terms = {**terms, "loss": terms["author_loss"] + auxiliary, "weighted_auxiliary": auxiliary}
    return terms, restored

def fit_step(model, optimizer, raw_clean, observed, mean, std, metric, arm, *, auxiliary_weight=.1,
             probe_auxiliary=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    truth_z = ((raw_clean - mean) / std).float().reshape(50, 128, 3)
    result = model(torch.where(observed.reshape(50, 128, 3), truth_z, 0.),
        observed.reshape(50, 128, 3), truth_z, stage="train")
    terms, restored = loss_terms(result, raw_clean, observed, mean, std, metric, arm, auxiliary_weight)
    active = [p for p in model.parameters() if p.requires_grad]
    auxiliary_norm = None
    if probe_auxiliary and arm != "base":
        gradients = torch.autograd.grad(terms["weighted_auxiliary"], active, retain_graph=True, allow_unused=True)
        squares = [g.detach().double().square().sum() for g in gradients if g is not None]
        auxiliary_norm = float(torch.stack(squares).sum().sqrt()) if squares else 0.
        if not np.isfinite(auxiliary_norm) or auxiliary_norm <= 0:
            raise FloatingPointError("finite nonzero auxiliary gradient required")
    terms["loss"].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active):
        raise FloatingPointError("nonfinite SAITS gradient")
    norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in active):
        raise FloatingPointError("nonfinite SAITS parameter")
    return {**{key: float(value.detach()) for key, value in terms.items()},
        "gradient_norm": float(norm), "auxiliary_gradient_norm": auxiliary_norm,
        "observed_values_preserved": bool(torch.equal(restored[observed], raw_clean[observed]))}

