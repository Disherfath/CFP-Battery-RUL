"""Exact strong-reference update/inference from paper repeatability experiments."""
import numpy as np
import torch
from . import t1_training as t1
from . import completion as saits

def train_step(kind, model, optimizer, raw_fit, masks_fit, plan, step, mean, std, *, target_fit_z=None):
    if kind == "t1_within":
        i, view = int(plan["cell"][step]), int(plan["view"][step])
        return t1.fit_step(model, optimizer, raw_fit[i:i+1], masks_fit[i:i+1, view], mean, std)
    model.train(); optimizer.zero_grad(set_to_none=True)
    ids = {k: torch.as_tensor(plan[k][step], device=raw_fit.device) for k in ("cell", "cycle", "view")}
    if target_fit_z is None or target_fit_z.dtype != torch.float32 or target_fit_z.shape != raw_fit.shape:
        raise ValueError("SAITS requires the exact original CPU-normalized float32 fit tensor")
    truth = target_fit_z[ids["cell"], ids["cycle"]]
    observed = masks_fit[ids["cell"], ids["view"], ids["cycle"]]
    result = model(truth, observed, truth, stage="train")
    torch.testing.assert_close(result["loss"], result["reconstruction_loss"]+result["imputation_loss"], rtol=0, atol=0)
    if not torch.isfinite(result["loss"]):
        raise FloatingPointError("nonfinite original SAITS objective")
    result["loss"].backward()
    active = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active):
        raise FloatingPointError("nonfinite SAITS gradient")
    norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in active):
        raise FloatingPointError("nonfinite SAITS parameter")
    return {"loss": float(result["loss"].detach()), "ort": float(result["reconstruction_loss"].detach()),
        "mit": float(result["imputation_loss"].detach()), "gradient_norm": float(norm)}

@torch.no_grad()
def complete_queries(kind, model, bank, masks, stats, device):
    if kind == "saits64":
        return saits.complete_queries(model, bank, masks, stats, device)
    model.eval()
    result = np.asarray(bank, dtype=np.float64).copy()
    if result.shape != masks.shape or result.shape[1:] != (10,50,128,3) or masks.dtype != bool:
        raise ValueError("canonical ten-view bank required")
    for i in range(len(result)):
        for view in range(1,10):
            raw = torch.as_tensor(bank[i:i+1,view], dtype=torch.float64, device=device)
            observed = torch.as_tensor(masks[i:i+1,view], dtype=torch.bool, device=device)
            result[i:i+1,view] = model.complete_raw(raw, observed).cpu().numpy()
    np.testing.assert_array_equal(result[masks], bank[masks])
    np.testing.assert_array_equal(result[:,0], bank[:,0])
    return result

