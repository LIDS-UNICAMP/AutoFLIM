"""Model-size / compute-efficiency measurement for the multi-objective GA.

Provides a single entry point, ``measure_efficiency``, that returns a
``(flops, n_params)`` pair for a trained FLIMModel. FLOPs is the preferred
size objective for FLIM encoders because the convolutions run at full image
resolution, so the real cost scales with H*W*kernels*ks^2 — a signal that
param count (resolution-blind) cannot see. Param count is always returned as a
dependency-free fallback/secondary readout.

FLOPs is measured with ``torch.utils.flop_counter.FlopCounterMode`` on a single
forward pass over a fixed sample input. FlopCounterMode intercepts aten ops in
*eager* mode (via __torch_dispatch__) and sums the cost of every conv/matmul it
sees — module or functional, encoder or decoder — exactly the operator coverage
fvcore had, but without any JIT trace.

Why not fvcore's ``FlopCountAnalysis`` (used previously): it computes FLOPs by
jit-tracing the model, and the FLIM forward contains data-dependent control flow
(``torch.unique``, ``if fg_count > 0``, boolean-mask indexing in the adaptive
decoder; a ``tensor -> bool`` branch in the conv layers). For some genomes that
trace fails inside ``torch::jit::FixupTraceScopeBlocks`` with a C++ INTERNAL
ASSERT that throws a ``c10::Error`` and calls ``std::terminate`` — an uncatchable
``SIGABRT`` that kills the whole search process (observed on schisto split3).
Other genomes left unpicklable scope-hook closures on the model that broke a
later ``torch.save`` (split2). FlopCounterMode runs the forward normally (the
data-dependent branches just execute), so neither failure mode can occur. Any
error still falls back to ``flops=None`` and the caller uses param count instead.
"""
from typing import Optional, Tuple

import torch


def count_params(model) -> int:
    """Total tensor elements in the model (parameters, with buffer fallback).

    FLIM learners store kernels as conv weights (nn.Parameter); some variants
    keep them as registered buffers. Count buffers when no parameters exist so
    the size signal is never silently zero.
    """
    n = int(sum(p.numel() for p in model.parameters()))
    if n == 0:
        n = int(sum(b.numel() for b in model.buffers()))
    return n


def measure_efficiency(
    model,
    sample_input: torch.Tensor,
) -> Tuple[Optional[int], int]:
    """Return ``(flops, n_params)`` for ``model``.

    ``sample_input`` is a representative input tensor (1, C, H, W) on the
    model's device; FLOPs depends on resolution, so callers must pass a fixed
    canonical shape to keep the metric comparable across individuals.

    flops is None when FlopCounterMode is unavailable or the forward pass
    cannot run; callers should fall back to n_params in that case.
    """
    n_params = count_params(model)

    flops: Optional[int] = None
    was_training = model.training
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model.eval()
        try:
            flop_counter = FlopCounterMode(display=False)
            with torch.no_grad(), flop_counter:
                model(sample_input)
            flops = int(flop_counter.get_total_flops())
        finally:
            if was_training:
                model.train()
    except Exception:
        flops = None

    return flops, n_params
