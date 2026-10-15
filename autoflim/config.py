"""
Configuration for AutoFLIM runs.

A run is fully described by one YAML file: where the dataset lives, which
architectures the search may consider, and how the genetic algorithm and its
inner greedy image-selection behave. Everything has a default except the
dataset paths, so a minimal config is a handful of lines.

The YAML maps 1:1 onto the dataclasses below::

    dataset:      -> DatasetConfig
    search_space: -> SearchSpaceConfig
    ga:           -> GAConfig
    greedy:       -> GreedyConfig
    markers:      -> MarkerConfig
    model:        -> ModelConfig
    eval:         -> EvalConfig
    runtime:      -> RuntimeConfig
    output:       -> OutputConfig

Unknown keys are rejected rather than ignored, so a typo in a config fails
loudly at load time instead of silently running the default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Allowed values. These mirror the choices the underlying modules accept; they
# are duplicated here so a bad config fails at load time with a clear message
# rather than deep inside a spawned worker process.
# ---------------------------------------------------------------------------

NETWORK_TYPES = ('standard', 'separable', 'convnext', 'depthwise')
ACTIVATIONS = ('relu', 'leaky', 'gelu', 'silu', 'abs', 'identity')
SKIP_PATTERNS = ('none', 'residual')
POOL_TYPES = (
    'avg_pool', 'avg_pool_s1', 'avg_pool_s2',
    'max_pool', 'max_pool_s1', 'max_pool_s2',
    'strided_conv', 'strided_conv_s1', 'strided_conv_s2',
    'none',
)
COMBINE_STRATEGIES = (
    'none', 'inferior_reconstruction', 'superior_reconstruction',
    'mean', 'sum', 'max',
)
LEARNERS = ('bofp', 'cluster')
HEAD_TYPES = ('ift', 'ca', 'none')
COLOR_SPACES = ('lab', 'gray')
BINARIZE_METHODS = ('otsu', 'otsu_local', 'yen', 'sauvola')
POST_PROCESSING = ('area_filter', 'largest_component', 'none')
GREEDY_STRATEGIES = (
    'worst_first', 'random', 'median_first', 'diverse',
    'fa_guided', 'first_quartile', 'second_quartile',
)
COLD_START_STRATEGIES = ('coreset', 'random', 'diverse')
DENSIFY_MODES = ('error_region', 'filter_analysis', 'random', 'none')
MUTATION_SCHEDULES = ('adaptive', 'static', 'linear')
SIZE_OBJECTIVES = ('none', 'nsga2')
EFFICIENCY_METRICS = ('flops', 'params')


class ConfigError(Exception):
    """Raised when a config file is malformed, or a value is out of range."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Where the images, ground truth and split CSVs live.

    The on-disk layout is the one `autoflim prepare` produces::

        <path>/<images_dir>/000001.png     # input images
        <path>/<labels_dir>/000001.png     # binary ground-truth masks
        <splits_dir>/split1/train_100pct.csv
        <splits_dir>/split1/val.csv
        <splits_dir>/split1/test.csv

    Split CSVs are headerless, one image filename per line.
    """

    name: str
    path: Path
    images_dir: str = 'images'
    labels_dir: str = 'truelabels'
    # Split CSVs. Defaults to `data/<name>` relative to the repo root.
    splits_dir: Optional[Path] = None
    splits: List[int] = field(default_factory=lambda: [1])
    # Which train_<pct>pct.csv to use, interpolated verbatim: '100' reads
    # train_100pct.csv. Keep the zero padding you passed to
    # `prepare --percentages` ('025', not '25') or the file will not be found.
    train_percentage: str = '100'
    image_ext: str = '.png'
    label_ext: str = '.png'

    @property
    def images_folder(self) -> Path:
        return self.path / self.images_dir

    @property
    def labels_folder(self) -> Path:
        return self.path / self.labels_dir

    def train_csv(self, split: int) -> Path:
        return self.splits_dir / f'split{split}' / f'train_{self.train_percentage}pct.csv'

    def val_csv(self, split: int) -> Path:
        return self.splits_dir / f'split{split}' / 'val.csv'

    def test_csv(self, split: int) -> Path:
        return self.splits_dir / f'split{split}' / 'test.csv'


@dataclass
class SearchSpaceConfig:
    """The architectures the GA may construct.

    Every field is a list of candidate values; the GA picks one per genome.
    A single-element list pins that gene, which is how you shrink the space.
    """

    kernel_sizes: List[Union[int, List[int]]] = field(default_factory=lambda: [3])
    n_layers: List[int] = field(default_factory=lambda: [2, 3])
    nkernels_per_marker: List[int] = field(default_factory=lambda: [1])
    noutput_channels: List[int] = field(default_factory=lambda: [500])
    network_types: List[str] = field(default_factory=lambda: ['standard'])
    activations: List[str] = field(default_factory=lambda: ['relu'])
    skip_patterns: List[str] = field(default_factory=lambda: ['none'])
    pool_types: List[str] = field(default_factory=lambda: ['avg_pool'])
    # FLIM kernel-init spread around cluster centres.
    stdev_factors: List[float] = field(default_factory=lambda: [0.01])
    # IFT erosion radius and marker adjacency radius are searchable too.
    erode_radii: List[float] = field(default_factory=lambda: [1.0])
    marker_adjacencies: List[float] = field(default_factory=lambda: [1.5])
    # Multi-layer saliency fusion. 'none' decodes the last layer only.
    combine_strategies: List[str] = field(default_factory=lambda: ['none'])

    def to_search_space(self, head_type: str):
        """Build the runtime `SearchSpace` used by the GA."""
        from .ga.genome import SearchSpace

        return SearchSpace(
            kernel_sizes=[tuple(k) if isinstance(k, list) else k
                          for k in self.kernel_sizes],
            n_layers_list=list(self.n_layers),
            nkernels_per_marker_list=list(self.nkernels_per_marker),
            noutput_channels_list=list(self.noutput_channels),
            network_types=list(self.network_types),
            erode_radius_list=list(self.erode_radii),
            marker_adj_list=list(self.marker_adjacencies),
            skip_patterns=list(self.skip_patterns),
            stdev_factor_list=list(self.stdev_factors),
            pool_type_list=list(self.pool_types),
            head_type_list=[head_type],
            # 'none' is the YAML spelling of "no fusion"; the runtime wants None.
            combine_strategy_list=[None if c == 'none' else c
                                   for c in self.combine_strategies],
            activation_list=list(self.activations),
        )

    def size(self) -> int:
        """Rough count of distinct architectures, for reporting."""
        per_layer = len(self.kernel_sizes) * len(self.pool_types)
        total = 0
        for n in self.n_layers:
            total += (per_layer ** n)
        return (total
                * len(self.nkernels_per_marker)
                * len(self.noutput_channels)
                * len(self.network_types)
                * len(self.activations)
                * len(self.skip_patterns)
                * len(self.stdev_factors)
                * len(self.erode_radii)
                * len(self.marker_adjacencies)
                * len(self.combine_strategies))


@dataclass
class GAConfig:
    """Genetic algorithm hyper-parameters."""

    population_size: int = 20
    n_generations: int = 20
    elite_k: int = 2
    tournament_size: int = 3
    # Per-gene mutation probability. Kept near 1/genome-length so crossover
    # heritability survives; `max_mutation_rate` caps every schedule.
    mutation_rate: float = 0.3
    min_mutation_rate: float = 0.05
    max_mutation_rate: float = 0.3
    mutation_schedule: str = 'adaptive'
    # UCB exploration weight for parent selection. 0 = exploit, 1 = explore.
    ucb_beta: float = 0.5
    # Stop once the global best has not improved for this many generations.
    # 0 runs all generations.
    best_stagnation_patience: int = 0
    # Re-score elites every other generation so their (arch, images, markers)
    # triple keeps pace with the architecture. Disable for true elitism.
    reeval_elites: bool = True
    # 'nsga2' searches the (Dice, size) Pareto front instead of Dice alone.
    size_objective: str = 'none'
    efficiency_metric: str = 'flops'
    # JSON-serialisable seed genome injected into generation 0.
    initial_genome: Optional[Dict[str, Any]] = None
    seed: Optional[int] = 123


@dataclass
class GreedyConfig:
    """Inner active-learning loop that picks training images per genome.

    Each genome is scored by greedily growing its training set: add the image
    the current model does worst on, retrain, keep the image if validation Dice
    improves by at least `eps`.
    """

    # Greedy steps for generation-0 individuals (cold start) and for offspring
    # (warm-started from the parent's image set).
    n_steps_init: int = 5
    n_steps_offspring: int = 5
    eps: float = 0.025
    patience: int = 3
    strategy: str = 'worst_first'
    cold_start_strategy: str = 'random'
    # Cap on candidates scored per step. -1 scores all; a value in (0, 1] is a
    # fraction of the remaining pool; > 1 is an absolute count.
    max_score_candidates: float = 50
    # Score each offspring from a fresh single seed image rather than
    # inheriting the parent's set, so fitness reflects architecture alone.
    no_image_inheritance: bool = False
    # Hard cap on images kept for final training (top-K by marginal gain).
    max_selected_images: int = -1


@dataclass
class MarkerConfig:
    """Initial marker counts and the marker-refinement (densify) policy."""

    n_fg_init: int = 5
    n_bg_init: int = 5
    # Place markers at FP/FN regions of a candidate before adding it.
    use_inference_guided: bool = True
    densify_mode: str = 'error_region'
    max_densify_steps: int = 3
    densify_threshold: float = 0.4
    densify_eps: float = 0.05
    densify_patience: int = 3
    densify_base_count: int = 3
    densify_scale: float = 1.5


@dataclass
class ModelConfig:
    """How FLIM models are built and trained."""

    learner: str = 'bofp'
    color_space: str = 'lab'
    # Segmentation head during the search. 'none' binarises the decoder output
    # with a threshold, which is much cheaper and fine for ranking.
    head_type: str = 'none'
    # Head for the single final model. Defaults to `head_type`.
    head_type_final: Optional[str] = None

    @property
    def final_head(self) -> str:
        return self.head_type_final if self.head_type_final is not None else self.head_type


@dataclass
class EvalConfig:
    """How predictions are turned into masks and scored."""

    binarize_method: str = 'yen'
    post_processing: str = 'area_filter'
    # Restrict the fitness set to images that actually contain foreground.
    # Without this, empty-GT images score Dice 1.0 for predicting nothing and
    # let under-segmenting architectures win the search.
    val_foreground_only: bool = False
    # Alternative to the above: keep empty-GT images but score them only as a
    # false-positive penalty. Ignored when val_foreground_only is set.
    val_empty_penalty: float = 0.0


@dataclass
class RuntimeConfig:
    """Parallelism, devices and logging."""

    gpu_ids: List[int] = field(default_factory=lambda: [0])
    # Splits run concurrently, each in its own process.
    num_workers: int = 1
    # Concurrent genome evaluations inside one split.
    num_workers_evaluation: int = 4
    # Cap on concurrent GPU trainings per split process. <= 0 is unlimited.
    max_gpu_concurrency: int = 4
    # Limit the training pool to this many images (debugging / quick runs).
    train_subset_size: Optional[int] = None
    tensorboard: bool = False
    # Log the best individual's marker images every N generations. 0 disables.
    tb_marker_log_every: int = 0
    resume: bool = False


@dataclass
class OutputConfig:
    """Where results, markers and models are written.

    Each run is namespaced by `exp_name`, so re-running with a new name never
    clobbers an earlier experiment.
    """

    exp_name: str = 'exp1'
    results_dir: Path = Path('results')
    markers_dir: Path = Path('markers')
    models_dir: Path = Path('models')
    test_results_dir: Path = Path('test_results')

    def run_results(self, dataset: str) -> Path:
        return self.results_dir / dataset / self.exp_name

    def run_markers(self, dataset: str) -> Path:
        return self.markers_dir / dataset / self.exp_name

    def run_models(self, dataset: str) -> Path:
        return self.models_dir / dataset / self.exp_name

    def run_test_results(self, dataset: str) -> Path:
        return self.test_results_dir / dataset / self.exp_name


@dataclass
class Config:
    """A complete AutoFLIM run configuration."""

    dataset: DatasetConfig
    search_space: SearchSpaceConfig = field(default_factory=SearchSpaceConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    markers: MarkerConfig = field(default_factory=MarkerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Path this config was loaded from, for provenance in saved run configs.
    source_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data view, suitable for yaml.safe_dump."""
        return _to_plain(dataclasses.asdict(self))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SECTIONS = {
    'dataset': DatasetConfig,
    'search_space': SearchSpaceConfig,
    'ga': GAConfig,
    'greedy': GreedyConfig,
    'markers': MarkerConfig,
    'model': ModelConfig,
    'eval': EvalConfig,
    'runtime': RuntimeConfig,
    'output': OutputConfig,
}

# Fields that are paths and should be expanded (~ and relative-to-config).
_PATH_FIELDS = {
    'dataset': ('path', 'splits_dir'),
    'output': ('results_dir', 'markers_dir', 'models_dir', 'test_results_dir'),
}


def _to_plain(obj: Any) -> Any:
    """Recursively convert Paths/tuples so yaml.safe_dump accepts the result."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _build_section(name: str, cls, raw: Dict[str, Any], base_dir: Path):
    """Instantiate one config dataclass from its YAML mapping."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{name}' must be a mapping, got {type(raw).__name__}.")

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Section '{name}' has unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(known))}."
        )

    kwargs = dict(raw)

    # Resolve path-typed fields relative to the config file's directory, so a
    # config is portable and can use relative paths.
    for key in _PATH_FIELDS.get(name, ()):
        if kwargs.get(key) is not None:
            kwargs[key] = _resolve_path(kwargs[key], base_dir)

    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"Section '{name}': {exc}") from exc


def _resolve_path(value: Any, base_dir: Path) -> Path:
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_config(path: Union[str, Path],
                overrides: Optional[Dict[str, Any]] = None,
                check_paths: bool = True) -> Config:
    """Load, validate and return a `Config`.

    `overrides` is a flat mapping of dotted keys applied after parsing, e.g.
    ``{'ga.n_generations': 5, 'runtime.gpu_ids': [0, 1]}``. This is how CLI
    flags beat the file without duplicating every option as an argument.

    `check_paths=False` skips the filesystem checks, so a config can be
    inspected without the dataset mounted.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections.")

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ConfigError(
            f"{path}: unknown top-level section(s): {', '.join(sorted(unknown))}. "
            f"Valid sections: {', '.join(sorted(_SECTIONS))}."
        )

    if 'dataset' not in raw:
        raise ConfigError(f"{path}: a 'dataset' section is required.")

    base_dir = path.parent.resolve()
    sections = {
        name: _build_section(name, cls, raw.get(name), base_dir)
        for name, cls in _SECTIONS.items()
    }

    cfg = Config(source_path=path, **sections)

    # `splits_dir` defaults to <repo root>/data/<name>. The repo root is the
    # parent of the configs/ directory the file came from.
    if cfg.dataset.splits_dir is None:
        repo_root = base_dir.parent if base_dir.name == 'configs' else base_dir
        cfg.dataset.splits_dir = repo_root / 'data' / cfg.dataset.name

    if overrides:
        apply_overrides(cfg, overrides)

    validate(cfg, check_paths=check_paths)
    return cfg


def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> None:
    """Apply dotted-key overrides in place."""
    for key, value in overrides.items():
        if value is None:
            continue
        section_name, _, field_name = key.partition('.')
        if not field_name:
            raise ConfigError(f"Override '{key}' must be of the form 'section.field'.")
        if section_name not in _SECTIONS:
            raise ConfigError(f"Override '{key}': unknown section '{section_name}'.")

        section = getattr(cfg, section_name)
        if not hasattr(section, field_name):
            raise ConfigError(
                f"Override '{key}': '{section_name}' has no field '{field_name}'."
            )

        if field_name in _PATH_FIELDS.get(section_name, ()):
            value = Path(str(value)).expanduser()

        setattr(section, field_name, value)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_choice(value, allowed, label: str, errors: List[str]) -> None:
    if value not in allowed:
        errors.append(f"{label}: '{value}' is not one of {', '.join(map(str, allowed))}.")


def _check_choices(values, allowed, label: str, errors: List[str]) -> None:
    if not values:
        errors.append(f"{label}: must not be empty.")
        return
    for v in values:
        _check_choice(v, allowed, label, errors)


def validate(cfg: Config, check_paths: bool = True) -> None:
    """Raise `ConfigError` listing every problem found.

    Collecting all errors before raising means one run surfaces every typo,
    instead of the user fixing them one failed launch at a time.
    """
    errors: List[str] = []
    ss, ga, gr, mk, md, ev, rt, ds = (
        cfg.search_space, cfg.ga, cfg.greedy, cfg.markers,
        cfg.model, cfg.eval, cfg.runtime, cfg.dataset,
    )

    # ---- enumerated values ----
    _check_choices(ss.network_types, NETWORK_TYPES, 'search_space.network_types', errors)
    _check_choices(ss.activations, ACTIVATIONS, 'search_space.activations', errors)
    _check_choices(ss.skip_patterns, SKIP_PATTERNS, 'search_space.skip_patterns', errors)
    _check_choices(ss.pool_types, POOL_TYPES, 'search_space.pool_types', errors)
    _check_choices(ss.combine_strategies, COMBINE_STRATEGIES,
                   'search_space.combine_strategies', errors)
    _check_choice(md.learner, LEARNERS, 'model.learner', errors)
    _check_choice(md.color_space, COLOR_SPACES, 'model.color_space', errors)
    _check_choice(md.head_type, HEAD_TYPES, 'model.head_type', errors)
    if md.head_type_final is not None:
        _check_choice(md.head_type_final, HEAD_TYPES, 'model.head_type_final', errors)
    _check_choice(ev.binarize_method, BINARIZE_METHODS, 'eval.binarize_method', errors)
    _check_choice(ev.post_processing, POST_PROCESSING, 'eval.post_processing', errors)
    _check_choice(gr.strategy, GREEDY_STRATEGIES, 'greedy.strategy', errors)
    _check_choice(gr.cold_start_strategy, COLD_START_STRATEGIES,
                  'greedy.cold_start_strategy', errors)
    _check_choice(mk.densify_mode, DENSIFY_MODES, 'markers.densify_mode', errors)
    _check_choice(ga.mutation_schedule, MUTATION_SCHEDULES, 'ga.mutation_schedule', errors)
    _check_choice(ga.size_objective, SIZE_OBJECTIVES, 'ga.size_objective', errors)
    _check_choice(ga.efficiency_metric, EFFICIENCY_METRICS, 'ga.efficiency_metric', errors)

    # ---- numeric ranges ----
    if ga.population_size < 2:
        errors.append('ga.population_size: must be >= 2.')
    if ga.n_generations < 1:
        errors.append('ga.n_generations: must be >= 1.')
    if not 0 <= ga.elite_k < ga.population_size:
        errors.append('ga.elite_k: must be in [0, ga.population_size).')
    if ga.tournament_size < 1 or ga.tournament_size > ga.population_size:
        errors.append('ga.tournament_size: must be in [1, ga.population_size].')
    for name, val in (('mutation_rate', ga.mutation_rate),
                      ('min_mutation_rate', ga.min_mutation_rate),
                      ('max_mutation_rate', ga.max_mutation_rate)):
        if not 0.0 <= val <= 1.0:
            errors.append(f'ga.{name}: must be in [0, 1].')
    if ga.min_mutation_rate > ga.max_mutation_rate:
        errors.append('ga.min_mutation_rate: must be <= ga.max_mutation_rate.')
    if ga.ucb_beta < 0:
        errors.append('ga.ucb_beta: must be >= 0.')

    if gr.n_steps_init < 1:
        errors.append('greedy.n_steps_init: must be >= 1.')
    if gr.n_steps_offspring < 1:
        errors.append('greedy.n_steps_offspring: must be >= 1.')
    if gr.eps < 0:
        errors.append('greedy.eps: must be >= 0.')
    if gr.max_score_candidates == 0:
        errors.append('greedy.max_score_candidates: must be -1 or > 0.')

    if mk.n_fg_init < 1 or mk.n_bg_init < 1:
        errors.append('markers.n_fg_init / n_bg_init: must be >= 1.')
    if not 0.0 <= ev.val_empty_penalty <= 1.0:
        errors.append('eval.val_empty_penalty: must be in [0, 1].')
    if ev.val_foreground_only and ev.val_empty_penalty > 0:
        errors.append(
            'eval.val_empty_penalty is ignored when eval.val_foreground_only is '
            'set (no empty val images remain); set only one of them.'
        )

    if not rt.gpu_ids:
        errors.append('runtime.gpu_ids: must list at least one GPU id.')
    if rt.num_workers < 1:
        errors.append('runtime.num_workers: must be >= 1.')
    if rt.num_workers_evaluation < 1:
        errors.append('runtime.num_workers_evaluation: must be >= 1.')

    if any(n < 1 for n in ss.n_layers):
        errors.append('search_space.n_layers: values must be >= 1.')
    if not ds.splits:
        errors.append('dataset.splits: must list at least one split.')

    # A residual skip pattern adds a skip every 2 layers, so it needs >= 2.
    if 'residual' in ss.skip_patterns and max(ss.n_layers, default=0) < 2:
        errors.append(
            "search_space.skip_patterns: 'residual' requires at least one "
            'n_layers value >= 2.'
        )

    # ---- filesystem ----
    if check_paths:
        if not ds.images_folder.is_dir():
            errors.append(f'dataset: images folder not found: {ds.images_folder}')
        if not ds.labels_folder.is_dir():
            errors.append(f'dataset: labels folder not found: {ds.labels_folder}')
        for split in ds.splits:
            for csv_path in (ds.train_csv(split), ds.val_csv(split)):
                if not csv_path.exists():
                    errors.append(
                        f'dataset: split CSV not found: {csv_path} '
                        f"(run 'autoflim prepare' to generate splits)"
                    )

    if errors:
        bullet = '\n  - '
        raise ConfigError(
            f"Invalid configuration"
            f"{f' ({cfg.source_path})' if cfg.source_path else ''}:"
            f"{bullet}{bullet.join(errors)}"
        )


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    """Write the fully resolved config next to a run's results, for provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict()
    data.pop('source_path', None)
    with open(path, 'w') as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
