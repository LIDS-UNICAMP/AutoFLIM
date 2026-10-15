"""Crash-safe checkpoint/resume for the GA encoder search.

The search can run for hours and is sometimes OOM-killed mid-way (a hard
SIGKILL leaves no traceback and no saved model, since the model is only written
after the whole search finishes). This module persists the full search state
once per generation so a re-run with ``--resume`` continues from the last
completed generation with the *same* population, image sets, candidate pools,
greedy step, evaluation cache and RNG stream — not a fresh restart.

What is captured each generation:
  - the population (genomes + selected image_set + scores + flops/n_params),
  - the cumulative best individual and (NSGA-II) the Pareto front,
  - the evaluation cache (so already-evaluated individuals are never recomputed),
  - the Python and NumPy RNG states (so the continuation draws the same
    crossover/mutation/selection sequence as an uninterrupted run would),
  - the per-mode counters (stagnation, mutation rate, boost) and the generation
    index, plus a config fingerprint to refuse resuming a mismatched run.

Writes are atomic (temp file + ``os.replace``) so a crash during the write can
never corrupt an existing checkpoint.
"""
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np

CKPT_VERSION = 1


def save_checkpoint(path, state: dict) -> None:
    """Atomically pickle ``state`` to ``path`` (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path) -> Optional[dict]:
    """Load a checkpoint, or return None if absent/unreadable/incompatible."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
    except Exception as exc:
        print(f"[checkpoint] failed to read {path}: {exc} — starting fresh")
        return None
    if state.get("version") != CKPT_VERSION:
        print(f"[checkpoint] version mismatch "
              f"({state.get('version')} != {CKPT_VERSION}) — starting fresh")
        return None
    return state


def capture_rng() -> dict:
    """Snapshot the global Python + NumPy RNG state for exact continuation."""
    return {"python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(rng_state: Optional[dict]) -> None:
    """Restore RNG state captured by :func:`capture_rng` (best-effort)."""
    if not rng_state:
        return
    try:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
    except Exception as exc:
        print(f"[checkpoint] could not restore RNG state: {exc}")


def config_fingerprint(**fields) -> dict:
    """Identifying fields used to refuse resuming a checkpoint from a different
    run configuration (e.g. changed population size, objective, or split)."""
    return dict(fields)


def config_matches(ckpt: dict, expected: dict) -> bool:
    """True iff the checkpoint's stored fingerprint matches ``expected``."""
    saved = ckpt.get("config", {})
    for k, v in expected.items():
        if saved.get(k) != v:
            print(f"[checkpoint] config differs on {k!r} "
                  f"(checkpoint={saved.get(k)!r} != current={v!r}) — starting fresh")
            return False
    return True
