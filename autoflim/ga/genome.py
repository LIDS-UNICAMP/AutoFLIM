from dataclasses import dataclass, field
from typing import List, Any, Dict
import random

import numpy as np


@dataclass
class SearchSpace:
    kernel_sizes:             List[Any]   # int or tuple of ints per layer
    n_layers_list:            List[int]
    nkernels_per_marker_list: List[int]
    noutput_channels_list:    List[int]
    network_types:            List[str]
    erode_radius_list:        List[float] = field(default_factory=lambda: [8.0])
    marker_adj_list:          List[float] = field(default_factory=lambda: [1.5])
    skip_patterns:            List[str]   = field(default_factory=lambda: ['none'])
    # FLIM training hyper-parameters included in the search space.
    # Defaults reproduce the original single-value behaviour so existing scripts
    # that do not set these arguments are unaffected.
    stdev_factor_list:        List[float] = field(default_factory=lambda: [0.00000001])
    pool_type_list:           List[str]   = field(default_factory=lambda: ['avg_pool'])
    use_marker_norm_list:     List[bool]  = field(default_factory=lambda: [False])
    head_type_list:           List[str]   = field(default_factory=lambda: ['ift'])
    combine_strategy_list:    List[Any]   = field(default_factory=lambda: [None])
    activation_list:          List[str]   = field(default_factory=lambda: ['relu'])

    @property
    def _max_layers(self) -> int:
        return max(self.n_layers_list)

    def as_dict(self) -> Dict[str, List[Any]]:
        d: Dict[str, List[Any]] = {}
        for i in range(1, self._max_layers + 1):
            d[f'kernel_size_{i}'] = self.kernel_sizes
        d['n_layers']            = self.n_layers_list
        d['nkernels_per_marker'] = self.nkernels_per_marker_list
        d['noutput_channels']    = self.noutput_channels_list
        d['network_type']        = self.network_types
        d['erode_radius']        = self.erode_radius_list
        d['marker_adj']          = self.marker_adj_list
        d['skip_pattern']        = self.skip_patterns
        d['stdev_factor']        = self.stdev_factor_list
        for i in range(1, self._max_layers + 1):
            d[f'pool_type_{i}'] = self.pool_type_list
        d['use_marker_norm']     = self.use_marker_norm_list
        d['head_type']           = self.head_type_list
        d['combine_strategy']    = self.combine_strategy_list
        d['activation']          = self.activation_list
        return d


def _options_index(options: list, val: Any) -> int:
    """Index of val in options, normalising list/tuple for kernel_size."""
    for i, opt in enumerate(options):
        if opt == val:
            return i
        if (isinstance(opt, (list, tuple)) and isinstance(val, (list, tuple))
                and list(opt) == list(val)):
            return i
    return -1


def encode_genome(genome: dict, space: SearchSpace) -> np.ndarray:
    """Concatenated one-hot encoding of a genome (for surrogate models)."""
    parts = []
    for key, options in space.as_dict().items():
        vec = np.zeros(len(options), dtype=np.float32)
        idx = _options_index(options, genome.get(key))
        if idx >= 0:
            vec[idx] = 1.0
        parts.append(vec)
    return np.concatenate(parts)


def decode_genome(vec: np.ndarray, space: SearchSpace) -> dict:
    """Inverse of encode_genome."""
    genome = {}
    offset = 0
    for key, options in space.as_dict().items():
        segment = vec[offset: offset + len(options)]
        genome[key] = options[int(np.argmax(segment))]
        offset += len(options)
    return genome


def canonicalize_genome(genome: dict) -> dict:
    """Null-out kernel_size genes for layers beyond n_layers.

    A genome with n_layers=1 has kernel_size_2, kernel_size_3, … that are
    never used by the model.  Without canonicalization, two genomes that are
    functionally identical can have different raw dicts (different junk-gene
    values), which:
      - bypasses the duplicate-guard in the GA loop
      - misses the eval cache (cache key includes all genes)
      - causes re-evaluation with fresh random markers → noisy fitness

    Setting unused kernel_size genes to None makes the cache key and duplicate
    check correctly treat functionally equivalent architectures as identical.
    """
    g = dict(genome)
    n = g.get('n_layers', 1)
    i = 1
    while f'kernel_size_{i}' in g:
        if i > n:
            g[f'kernel_size_{i}'] = None
        i += 1
    i = 1
    while f'pool_type_{i}' in g:
        if i > n:
            g[f'pool_type_{i}'] = None
        i += 1
    return g


def is_valid_genome(genome: dict, space: SearchSpace) -> bool:
    """True iff every gene value is present in the search space.

    Layer-indexed genes (kernel_size_i / pool_type_i) for layers BEYOND the
    genome's n_layers are unused and conventionally None (see
    canonicalize_genome); they are skipped here so a valid shallow genome is not
    rejected for its None padding.

    Without this skip, ANY genome with n_layers < max(n_layers_list) fails
    validation — None is never a search-space option — which made crossover
    between two equally-shallow parents impossible: the child shares the parents'
    None in the deepest slots and was always rejected. Validity is now checked
    only over the layers the genome actually uses.
    """
    n_layers = genome.get('n_layers')
    for key, options in space.as_dict().items():
        if (n_layers is not None
                and key.startswith(('kernel_size_', 'pool_type_'))
                and int(key.rsplit('_', 1)[1]) > n_layers):
            continue
        if _options_index(options, genome.get(key)) < 0:
            return False
    return True


def random_genome(space: SearchSpace) -> dict:
    return canonicalize_genome(
        {key: random.choice(options) for key, options in space.as_dict().items()}
    )


def median_genome(space: SearchSpace) -> dict:
    return {key: options[len(options) // 2] for key, options in space.as_dict().items()}
