"""The paper's ReadPrune/FRA-Q2 selector; no repair or token aggregation."""

from __future__ import annotations

import numpy as np

from . import _math


def select_tokens(features, response_deltas, k):
    """Select exactly ``k`` original visual tokens in their original order.

    Args:
        features: Finite, nonzero-row ``[N, D]`` NumPy array or PyTorch tensor
            containing the native projected features used by the decoder.
        response_deltas: Signed ``[N]`` readout-alignment changes, i.e.
            ``(e_full - e_ablated) @ question_embedding``. The response vectors
            and question embedding must come from the paired frozen CLIP space.
        k: Integer budget in ``[1, N]``; this counts visual patch tokens only.

    Returns:
        A dictionary with sorted NumPy ``indices``, gathered ``tokens`` (same
        framework, device, and dtype as ``features``), raw ``quality`` weights,
        and numerical ``diagnostics``. Neither input is modified.

    This is a CPU FP64 reference selector. It does not by itself integrate with
    a VLM or guarantee a latency speedup. The quality is squared exactly once:
    ``L = diag(q**2) @ B @ diag(q**2)``. No exchange repair or NDR is applied.
    """
    B = _math.geometry(features)
    q, quality_diagnostics = _math.quality(response_deltas)
    if q.shape != (len(B),):
        raise ValueError("response_deltas must have one entry per visual token")
    selected = _math.select(B, np.square(q), k)
    indices = np.asarray(selected["support"], dtype=np.int64)
    if hasattr(features, "detach"):
        import torch

        device_indices = torch.as_tensor(indices, dtype=torch.long, device=features.device)
        tokens = features.index_select(0, device_indices)
    else:
        tokens = np.asarray(features)[indices]
    diagnostics = {
        **selected,
        "method": "ReadPrune",
        "legacy_method": "FRA_Q2",
        "quality": quality_diagnostics,
        "quality_exponent": 2,
        "quality_log_coefficient": 4,
        "repair_applied": False,
        "token_values_unchanged": True,
    }
    return {"indices": indices, "tokens": tokens, "quality": q,
            "diagnostics": diagnostics}
