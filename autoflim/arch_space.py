import yaml

# Number of input channels the encoder is built for. Must match how images are
# loaded: LAB/RGB → 3, grayscale ('L') → 1. Derived from color_space via
# set_input_channels(); defaults to 3 for backward compatibility.
_NINPUT_CHANNELS = 3


def set_input_channels(color_space):
    """Set the encoder's input-channel count from the dataset color space.

    'lab' loads images as 3-channel LAB; anything else ('gray') loads as a
    single-channel 'L' image (see pil_mode in greedy.py/selection.py). The arch
    must be built with the matching channel count, otherwise the FLIM learner
    indexes non-existent input channels ("index 1 is out of bounds for
    dimension 0 with size 1").

    Must be called inside each spawned subprocess (module globals do not inherit
    across the 'spawn' start method).
    """
    global _NINPUT_CHANNELS
    _NINPUT_CHANNELS = 3 if color_space == 'lab' else 1


# Process-wide encoder learner. 'bofp' or 'cluster'. Drives both the per-block
# learner type and the matching normalization settings in build_arch_from_genome:
#   bofp    → use_bias=True,  use_marker_based_normalization=False
#   cluster → use_bias=False, use_marker_based_normalization=True
_LEARNER_TYPE = 'cluster'


def set_learner_type(learner_type):
    """Select the encoder learner used for all subsequently-built archs.

    Must be called inside each spawned subprocess (module globals do not inherit
    across the 'spawn' start method).
    """
    if learner_type not in ('bofp', 'cluster'):
        raise ValueError(
            f"Unknown learner type: {learner_type!r} (expected 'bofp' or 'cluster')")
    global _LEARNER_TYPE
    _LEARNER_TYPE = learner_type


def _learner():
    return {"type": _LEARNER_TYPE, "clusterer": "faiss_kmeans"}

def _pool(base_type, stride):
    return {"type": base_type, "size": [3, 3], "stride": stride}


def _resolve_pool(pool_type):
    """Return (base_type, stride) for pool-based pool_types.

    Supports legacy names (avg_pool, max_pool → stride 2) and explicit
    stride variants (avg_pool_s1, avg_pool_s2, max_pool_s1, max_pool_s2).
    """
    if pool_type.endswith('_s1'):
        return pool_type[:-3], 1
    if pool_type.endswith('_s2'):
        return pool_type[:-3], 2
    return pool_type, 2  # legacy default


class ArchDumper(yaml.SafeDumper):
    """SafeDumper that suppresses aliases and writes plain lists inline."""
    def ignore_aliases(self, data):
        return True

    def represent_sequence(self, tag, sequence, flow_style=None):
        if any(isinstance(i, dict) for i in sequence):
            return super().represent_sequence(tag, sequence, flow_style=False)
        return super().represent_sequence(tag, sequence, flow_style=True)


def dump_arch(arch_cfg, stream):
    yaml.dump(arch_cfg, stream, Dumper=ArchDumper, default_flow_style=False,
              sort_keys=False, allow_unicode=True)


_STRIDED_CONV_TYPES = {'strided_conv', 'strided_conv_s1', 'strided_conv_s2'}

# Label abbreviations for per-layer pool_type encoding in arch labels.
_PT_ABBREV = {
    'avg_pool':       'avg',   # legacy → stride 2
    'avg_pool_s1':    'avg1',
    'avg_pool_s2':    'avg2',
    'max_pool':       'max',   # legacy → stride 2
    'max_pool_s1':    'max1',
    'max_pool_s2':    'max2',
    'none':           'none',
    'strided_conv':   'sc',    # legacy → stride 2
    'strided_conv_s1': 'sc1',
    'strided_conv_s2': 'sc2',
}


def _pt_abbrev(pt):
    return _PT_ABBREV.get(pt, pt.split('_')[0] if pt else 'none')


def _make_block(name, inputs, conv_type, ks, nkpm, noc, pool_type='avg_pool', activation='relu'):
    """Build a single block dict."""
    is_strided_conv = pool_type in _STRIDED_CONV_TYPES
    no_pool         = pool_type is None or pool_type == 'none'
    # strided_conv and strided_conv_s2 → stride 2; strided_conv_s1 → stride 1
    conv_stride = 2 if pool_type in ('strided_conv', 'strided_conv_s2') else 1
    if no_pool or is_strided_conv:
        pooling = None
    else:
        base_pt, pool_stride = _resolve_pool(pool_type)
        pooling = _pool(base_pt, pool_stride)
    block = {
        "name":                name,
        "conv_type":           conv_type,
        "block_type":          "conv",
        "inputs":              inputs,
        "kernel_size":         [ks, ks],
        "nkernels_per_marker": nkpm,
        "noutput_channels":    noc,
        "dilation_values":     [1, 1],
        "stride":              conv_stride,
        "padding_mode":        "reflect",
        "activation":          activation,
        "learnable":           True,
        "learner":             _learner(),
        "pooling":             pooling,
    }
    if conv_type == "convnext":
        block["kwargs"]     = {"expand_ratio": 4}
        block["activation"] = "identity"  # GELU is internal to ConvNextBlock
    return block


def _make_skip_add_block(name, inputs):
    return {
        "name":       name,
        "block_type": "skip_add",
        "inputs":     inputs,
        "learnable":  False,
        "learner":    None,
    }


def _build_head_cfg(head_type: str, erode_radius: float):
    """Return the head config dict for the given head_type, or None for 'none'."""
    if head_type == 'ca':
        return {"type": "ca_gpu_segmentation", "kwargs": {"erode_radius": erode_radius}}
    if head_type == 'none':
        return None
    return {"type": "ift_segmentation", "kwargs": {"erode_radius": erode_radius}}



def genome_label(genome):
    nlayers = genome['n_layers']
    ks_str = "-".join(str(genome[f'kernel_size_{i}']) for i in range(1, nlayers + 1))
    er   = genome.get('erode_radius', 8.0)
    adj  = genome.get('marker_adj', 1.5)
    sp   = genome.get('skip_pattern', 'none')
    label = (
        f"nl{genome['n_layers']}_ks{ks_str}_"
        f"nkpm{genome['nkernels_per_marker']}_noc{genome['noutput_channels']}_"
        f"nt{genome['network_type']}_er{er}_adj{adj}"
        f"_sk{sp}"
    )
    # Optional genes appended only when non-default, preserving backward compat
    # with cached labels from runs that predate these search dimensions.
    sf = genome.get('stdev_factor')
    if sf is not None and sf != 0.01:
        label += f"_sf{sf:.3g}"
    if f'pool_type_1' in genome:
        pts = [genome.get(f'pool_type_{i}') or 'avg_pool' for i in range(1, nlayers + 1)]
        if any(pt != 'avg_pool' for pt in pts):
            label += "_pt" + "-".join(_pt_abbrev(pt) for pt in pts)
    else:
        pt = genome.get('pool_type', 'avg_pool')
        if pt != 'avg_pool':
            label += f"_pt{pt.split('_')[0] if pt else 'none'}"
    if genome.get('use_marker_norm'):
        label += "_mn1"
    ht = genome.get('head_type')
    if ht is not None and ht != 'ift':
        label += f"_ht{ht}"
    cs = genome.get('combine_strategy')
    if cs is not None:
        label += f"_cs{cs}"
    act = genome.get('activation', 'relu')
    if act != 'relu':
        label += f"_act{act}"
    return label


def marker_snapshot_subdir(genome, image_set):
    """Unique snapshot subdir name for a (genome, image_set) pair.

    genome_label alone collides when two individuals share a genome but hold
    different training image sets: one individual's adaptive markers would
    overwrite the other's in the shared snapshot dir, so the markers reused for
    the best individual's final training could come from a different run.
    Appending a short hash of the sorted image set disambiguates them.
    """
    import hashlib
    h = hashlib.md5("|".join(sorted(image_set)).encode()).hexdigest()[:8]
    return f"{genome_label(genome)}__{h}"


def build_arch_from_genome(genome, erode_radius=None):
    """Return (arch_cfg dict, arch_label str) for a given genome.

    The returned dict is compatible with ArchitectureConfig.from_yaml() —
    write it with yaml.safe_dump, then load with ArchitectureConfig.from_yaml().
    erode_radius is forwarded to the IFT segmentation head.

    Supported network_type values: standard, separable, convnext.
    - convnext: adds kwargs={expand_ratio:4} and activation=identity per block.

    kernel_size in the genome can be an int (same for all layers) or a list of
    ints with length == n_layers (one kernel size per layer).

    Searchable genome fields beyond the core architecture:
      stdev_factor     (float, default 0.01)  — BOFP kernel init spread around
                       cluster centres; dataset-sensitive, search [0.005, 0.01, 0.05].
      pool_type        (str,   default 'avg_pool') — 'avg_pool' or 'max_pool'.
      use_marker_norm  (bool,  default False) — marker-based layer normalisation;
                       useful when training on multiple images with different intensity
                       distributions (the typical multi-image greedy refinement case).

    skip_pattern='residual' with odd n_layers: the final unpaired conv has no
    skip connection — it receives the preceding skip_add as input and feeds the
    decoder directly. This is architecturally valid; the decoder always reads
    from the last conv block regardless of whether it participates in a skip pair.
    """
    nlayers    = genome['n_layers']
    nkpm       = genome['nkernels_per_marker']
    noc        = genome['noutput_channels']
    conv_type  = genome['network_type']  # standard | separable | convnext | depthwise
    er         = float(erode_radius) if erode_radius is not None else float(genome.get('erode_radius', 8.0))
    activation = genome.get('activation', 'relu')
    label      = genome_label(genome)

    ks_per_layer = [genome[f'kernel_size_{i}'] for i in range(1, nlayers + 1)]
    # Per-layer pool types; fall back to global 'pool_type' for old-style genomes
    if f'pool_type_1' in genome:
        pt_per_layer = [genome.get(f'pool_type_{i}') or 'avg_pool' for i in range(1, nlayers + 1)]
    else:
        _global_pt = genome.get('pool_type', 'avg_pool')
        pt_per_layer = [_global_pt] * nlayers

    block_noc = noc

    skip_pattern = genome.get('skip_pattern', 'none')
    blocks = []

    if skip_pattern == 'none' or nlayers == 1:
        for i in range(1, nlayers + 1):
            inp = ["input"] if i == 1 else [f"conv{i-1}"]
            blocks.append(_make_block(f"conv{i}", inp, conv_type, ks_per_layer[i-1], nkpm, block_noc, pt_per_layer[i-1], activation))
        last_name = f"conv{nlayers}"

    elif skip_pattern == 'residual':
        # ResNet-style: skip_add closes every pair of consecutive conv layers.
        # last_skip_name tracks the entry point for the current skip pair.
        # Starts from conv1 (not "input") so skip_add never shortcuts to the raw image.
        # For odd n_layers the final unpaired conv receives the last skip_add as input
        # — see docstring for why this is valid.
        last_skip_name = "conv1"
        for i in range(1, nlayers + 1):
            if i == 1:
                inp = ["input"]
            elif i % 2 == 1:
                inp = [last_skip_name]
            else:
                inp = [f"conv{i-1}"]
            blocks.append(_make_block(f"conv{i}", inp, conv_type, ks_per_layer[i-1], nkpm, block_noc, pt_per_layer[i-1], activation))
            if i % 2 == 0:
                skip_name = f"skip_add_{i}"
                # conv output is main (larger channels), shortcut is skip (padded up to match)
                blocks.append(_make_skip_add_block(skip_name, [f"conv{i}", last_skip_name]))
                last_skip_name = skip_name
        # Always decode from the last conv, which has marker_labels set by BOFP.
        # skip_add blocks are intermediate merge nodes and never have marker_labels.
        last_name = f"conv{nlayers}"

    else:
        raise ValueError(f"Unknown skip_pattern: {skip_pattern!r}")

    combine_strategy = genome.get('combine_strategy', None)
    # All conv blocks ordered shallow→deep; only used when strategy is set.
    all_conv_names = [f"conv{i}" for i in range(1, nlayers + 1)]
    decoder_layers = all_conv_names if combine_strategy is not None else [last_name]

    cfg = {
        "name":                           label,
        "stdev_factor":                   float(genome.get('stdev_factor', 0.01)),
        "network_type":                   "regular",
        "use_bias":                        _LEARNER_TYPE == 'bofp',
        "use_marker_based_normalization":  _LEARNER_TYPE == 'cluster',
        "ninput_channels":                 _NINPUT_CHANNELS,
        "blocks":                          blocks,
        "decoder": {
            "type":             "mean_based_adaptive_decoder",
            "adaptation_fn":    "robust_weights",
            "inputs":           [],
            "decoder_layers":   decoder_layers,
            "combine_strategy": combine_strategy,
            # Softens the coarse deep-grid floor in inferior_reconstruction fusion.
            # Ignored by other strategies. Not exposed as a flag; fixed default.
            "deep_blur_sigma":  1.5,
        },
        "head": _build_head_cfg(genome.get('head_type', 'ift'), er),
    }
    return cfg, label
