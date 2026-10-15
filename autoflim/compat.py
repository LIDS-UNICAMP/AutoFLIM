"""
Backwards compatibility for models pickled by the pre-package layout.

`torch.save(model, ...)` pickles by module path. Models trained before this
code became the `autoflim` package recorded their classes under flat top-level
names (`utils`, `arch_space`, ...), because the search ran with `src/` on
`sys.path`. Unpickling those checkpoints here would fail with
ModuleNotFoundError.

`install_legacy_module_aliases()` registers the old names in `sys.modules` as
aliases of their new homes, so `torch.load` resolves them. New checkpoints are
unaffected — they already carry `autoflim.*` paths.
"""

import sys

# old flat module name -> new dotted module name
_ALIASES = {
    "utils": "autoflim.utils",
    "arch_space": "autoflim.arch_space",
    "evaluation": "autoflim.evaluation",
}

_installed = False


def install_legacy_module_aliases() -> None:
    """Make pre-package checkpoints loadable. Idempotent."""
    global _installed
    if _installed:
        return

    import importlib

    for old_name, new_name in _ALIASES.items():
        # Never shadow a real module that is already imported (e.g. a
        # third-party `utils`): only fill in names nothing else claims.
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except ImportError:
            pass

    _installed = True
