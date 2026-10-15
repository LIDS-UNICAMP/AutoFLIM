"""
AutoFLIM — neural architecture search for FLIM encoders.

FLIM (Feature Learning from Image Markers) builds convolutional encoders from a
handful of user-drawn markers instead of backpropagation. AutoFLIM searches the
encoder architecture *and* the set of annotated training images jointly, with a
memetic genetic algorithm: each candidate architecture is scored by greedy
active-learning over the image pool, so an architecture is judged together with
the annotation budget it actually needs.

Typical use is through the CLI::

    autoflim prepare --images ... --labels ... --name mydata
    autoflim search  --config configs/mydata.yaml
    autoflim test    --config configs/mydata.yaml

The Python API mirrors it::

    from autoflim.config import load_config
    from autoflim.search import run_search

    cfg = load_config("configs/mydata.yaml")
    run_search(cfg)
"""

from ._version import __version__

__all__ = ["__version__"]
