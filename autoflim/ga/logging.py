import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .evolution import GenerationResult


def log_generation(result: 'GenerationResult', output_folder: Optional[Path]) -> None:
    record = {
        'timestamp':  datetime.utcnow().isoformat(),
        'generation': result.generation,
        'best_genome': result.best_genome,
        'best_mean':  round(result.best_mean, 4),
        'best_std':   round(result.best_std, 4),
        'pop_means':  [round(m, 4) for m in result.means],
        'pop_stds':   [round(s, 4) for s in result.stds],
        'cache_hits': result.n_cache_hits,
        'diversity':  round(float(np.std(result.means)), 4),
    }

    print(
        f"[gen={result.generation:02d}] "
        f"best={result.best_mean:.4f}±{result.best_std:.4f} | "
        f"genome={result.best_genome} | "
        f"diversity={record['diversity']:.4f} | "
        f"cache_hits={result.n_cache_hits}"
    )

    if output_folder is not None:
        log_path = output_folder / 'ga_log.jsonl'
        with open(log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


def log_genome_eval(genome: dict, mean: float, std: float, source: str) -> None:
    """source: 'fresh' | 'cache' | 'elite'"""
    print(
        f"  [eval/{source}] "
        f"genome={genome} -> Dice={mean:.4f}±{std:.4f}"
    )


# ---------------------------------------------------------------------------
# Per-evaluation CSV logger (thread-safe)
# ---------------------------------------------------------------------------

_EVAL_LOG_COLUMNS = [
    'wall_time_s',   # seconds since EvalLogger was created
    'generation',    # GA generation index (0 = init)
    'genome',        # genome_label string
    'genome_json',   # full genome as JSON (for exact reconstruction)
    'step',          # greedy step index ('baseline', 0, 1, 2, …)
    'outcome',       # 'baseline' | 'kept' | 'dropped'
    'n_train_images',
    'mean_score',
    'std_score',
]


class EvalLogger:
    """Thread-safe CSV logger that records every model trained during the GA.

    One row per trained model: wall time, genome, greedy step, Dice score.
    Designed to produce data for a scatter plot of accuracy vs. wall time.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._t0   = time.monotonic()
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(_EVAL_LOG_COLUMNS)

    def log(
        self,
        generation: int,
        genome: dict,
        step,           # 'baseline' | int
        outcome: str,   # 'baseline' | 'kept' | 'dropped'
        n_train_images: int,
        mean_score: float,
        std_score: float,
    ) -> None:
        from ..arch_space import genome_label as _genome_label
        row = [
            f"{time.monotonic() - self._t0:.3f}",
            generation,
            _genome_label(genome),
            json.dumps(genome, sort_keys=True),
            step,
            outcome,
            n_train_images,
            f"{mean_score:.6f}",
            f"{std_score:.6f}",
        ]
        with self._lock:
            with open(self._path, 'a', newline='') as f:
                csv.writer(f).writerow(row)


# ---------------------------------------------------------------------------
# TensorBoard logger
# ---------------------------------------------------------------------------

# RGB colors per class index (background=blue, foreground=red, others cycling)
_CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0,   0,   255),   # class 0 — background
    (255, 0,   0),     # class 1 — foreground
    (0,   200, 0),     # class 2
    (255, 200, 0),     # class 3
    (200, 0,   200),   # class 4
]

# Visualization size (pixels) for logged images — keeps the grid manageable.
_VIS_SIZE = 256
# Fixed screen-space marker radius in the resized image.  4 px is clearly
# visible even with many overlapping superpixel markers.
_MARKER_RADIUS = 4


def _draw_markers_screen(
    img: np.ndarray,
    seeds: List[Tuple[int, int, int]],
    radius: int = _MARKER_RADIUS,
) -> np.ndarray:
    """Draw filled circles of fixed *screen-space* radius on a resized image.

    ``seeds`` must already be in the coordinate space of ``img`` (scaled from
    the original image).  Using a fixed pixel radius ensures markers are always
    visible regardless of the original image dimensions.
    """
    vis = img.copy()
    H, W = vis.shape[:2]
    r_sq = radius * radius
    # Build a small disk offset table once.
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= r_sq
    ]
    for (x, y, lbl) in seeds:
        color = _CLASS_COLORS[int(lbl) % len(_CLASS_COLORS)]
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                vis[ny, nx] = color
    return vis


def _fmt_gene_value(val) -> str:
    """Compact, stable string key for a numeric gene value.

    Integers render without a decimal point; floats use up to 4 significant
    digits so that e.g. 0.01 / 0.005 / 0.05 stay distinct and short.
    """
    f = float(val)
    if f == int(f):
        return str(int(f))
    return f"{f:.4g}"


def _gene_value_sort_key(s: str):
    """Sort gene-value labels numerically when possible, else lexically."""
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


class TBLogger:
    """TensorBoard logger for the GA encoder search.

    Logs per-generation scalars (fitness, diversity, genome genes, mutation
    rate) and, via :meth:`log_marker_refinement`, the before/after evolution of
    each refined image's markers.

    Parameters
    ----------
    log_dir : Path
        Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter
        # flush_secs=5 ensures data appears in TensorBoard quickly without
        # waiting for explicit close().
        self._writer = SummaryWriter(str(log_dir), flush_secs=5)
        self._lock = threading.Lock()
        self._eval_count       = 0   # global counter across all individual evaluations
        self._refinement_count = 0   # global counter for marker-refinement events
        # Per-gene population history: gene -> {generation -> {value -> count}}.
        # Used to render readable convergence heatmaps in place of TB histograms.
        self._gene_hist: dict = {}

    # ------------------------------------------------------------------
    # Checkpoint support: persist/restore the in-memory gene-convergence
    # history so resumed runs render heatmaps over the full generation range
    # instead of restarting accumulation at the resume point.
    # ------------------------------------------------------------------
    def export_state(self) -> dict:
        """Return a picklable snapshot of the accumulated heatmap history."""
        with self._lock:
            return {
                'gene_hist': {
                    gene: {gen: dict(counts) for gen, counts in hist.items()}
                    for gene, hist in self._gene_hist.items()
                },
            }

    def import_state(self, state: Optional[dict]) -> None:
        """Restore a snapshot produced by :meth:`export_state` (no-op if empty)."""
        if not state:
            return
        with self._lock:
            self._gene_hist = state.get('gene_hist', {}) or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_generation(
        self,
        gen: int,
        population: list,
        best_ind,           # cumulative best Individual
        current_best,       # this generation's best Individual
        mutation_rate: float,
        effective_beta: float,
        *,
        # ---- Tier 2: convergence & diversity dynamics ----
        best_stagnation: Optional[int] = None,
        pop_stagnation: Optional[int] = None,
        n_injected: int = 0,
        n_elite_reverted: int = 0,
        mutation_boosted: bool = False,
        # ---- Tier 3: compute efficiency ----
        n_evals: Optional[int] = None,
        n_cache_hits: Optional[int] = None,
        gen_wall_time_s: Optional[float] = None,
        cumulative_wall_time_s: Optional[float] = None,
        # ---- Tier 4: multi-objective (NSGA-II) ----
        pareto_front: Optional[list] = None,
        efficiency_metric: str = 'flops',
    ) -> None:
        """Log scalars for one generation (marker evolution is logged separately
        via log_marker_refinement)."""
        means = [ind.mean_score for ind in population]

        with self._lock:
            w = self._writer

            # ---- Fitness: this generation's best vs. cumulative best ----
            # (gen_best == max(population means); no separate population/max.)
            w.add_scalar("fitness/gen_best",        current_best.mean_score, gen)
            w.add_scalar("fitness/cumulative_best", best_ind.mean_score,     gen)

            # ---- Population spread ----
            w.add_scalar("population/mean",      float(np.mean(means)), gen)
            w.add_scalar("population/min",       float(np.min(means)),  gen)
            w.add_scalar("population/diversity", float(np.std(means)),  gen)

            # ---- Best individual: training set size ----
            w.add_scalar("best/n_train_images", len(best_ind.image_set), gen)

            # ---- Best individual: model size / compute ----
            # Logged whenever measured (any run with efficiency measurement, not
            # just NSGA-II) so the accuracy/size trade-off of the shipped model is
            # always visible.
            if getattr(best_ind, "flops", None) is not None:
                w.add_scalar("best/flops", float(best_ind.flops), gen)
            if getattr(best_ind, "n_params", None) is not None:
                w.add_scalar("best/n_params", float(best_ind.n_params), gen)

            # ---- Per-image score spread of the cumulative best ----
            if best_ind.per_image_scores:
                scores = list(best_ind.per_image_scores.values())
                w.add_scalar("best/per_image_min",  float(np.min(scores)),  gen)
                w.add_scalar("best/per_image_mean", float(np.mean(scores)), gen)

            # ---- Search hyperparameters ----
            w.add_scalar("params/mutation_rate", mutation_rate,   gen)
            w.add_scalar("params/ucb_beta",      effective_beta,  gen)

            # ---- Genome genes (numeric only) of cumulative best ----
            for key, val in best_ind.genome.items():
                if isinstance(val, (int, float)):
                    w.add_scalar(f"genome/{key}", float(val), gen)

            # ==================================================================
            # Tier 2 — convergence & diversity dynamics
            # ==================================================================
            # Genotypic diversity: how many distinct architectures survive. A
            # collapse toward 1 signals premature convergence even while the
            # *fitness* spread (population/diversity) still looks healthy.
            genome_keys = {json.dumps(ind.genome, sort_keys=True)
                           for ind in population}
            w.add_scalar("convergence/unique_genomes", len(genome_keys), gen)

            # Per-gene value distribution across the whole population, rendered
            # as a convergence heatmap (generation x gene-value, cell = fraction
            # of population). This replaces TB's native histograms, which smooth
            # the low-cardinality discrete genes into unreadable ridgelines.
            # Both numeric and categorical (str/bool) genes are tracked so that
            # network_type / pool_type / skip_pattern convergence is visible too.
            for ind in population:
                for k, val in ind.genome.items():
                    if isinstance(val, bool):
                        key = str(val)
                    elif isinstance(val, (int, float)):
                        key = _fmt_gene_value(val)
                    elif isinstance(val, str):
                        key = val
                    else:
                        continue
                    counts = self._gene_hist.setdefault(k, {}).setdefault(gen, {})
                    counts[key] = counts.get(key, 0) + 1
            self._render_gene_heatmaps(gen)

            # Stagnation counters — plateaus become visible and line up with the
            # event markers below.
            if best_stagnation is not None:
                w.add_scalar("convergence/best_stagnation", best_stagnation, gen)
            if pop_stagnation is not None:
                w.add_scalar("convergence/pop_stagnation", pop_stagnation, gen)

            # Event markers (counts per generation) — overlay on the fitness
            # curve to explain why it jumped or stalled.
            w.add_scalar("events/n_injected",       n_injected,            gen)
            w.add_scalar("events/n_elite_reverted", n_elite_reverted,      gen)
            w.add_scalar("events/mutation_boosted", int(mutation_boosted), gen)

            # ==================================================================
            # Tier 3 — compute efficiency
            # ==================================================================
            if n_evals is not None:
                w.add_scalar("compute/n_evals", n_evals, gen)
                if n_cache_hits is not None and n_evals > 0:
                    w.add_scalar("compute/cache_hit_rate",
                                 n_cache_hits / n_evals, gen)
            if n_cache_hits is not None:
                w.add_scalar("compute/n_cache_hits", n_cache_hits, gen)
            if gen_wall_time_s is not None:
                w.add_scalar("compute/gen_wall_time_s", gen_wall_time_s, gen)
            if cumulative_wall_time_s is not None:
                w.add_scalar("compute/cumulative_wall_time_s",
                             cumulative_wall_time_s, gen)

            # ==================================================================
            # Tier 4 — multi-objective Pareto front (NSGA-II mode only)
            # ==================================================================
            if pareto_front:
                metric = efficiency_metric
                sizes = [p.size_cost(metric) for p in pareto_front]
                sizes = [s for s in sizes if s != float('inf')]
                dices = [p.mean_score for p in pareto_front]
                w.add_scalar("pareto/front_size", len(pareto_front), gen)
                w.add_scalar("pareto/best_dice", max(dices), gen)
                if sizes:
                    # The smallest model on the front and the size of the most
                    # accurate one — the two extremes of the trade-off curve.
                    w.add_scalar("pareto/min_size", min(sizes), gen)
                    w.add_scalar("pareto/max_size", max(sizes), gen)
                    best_dice_p = max(pareto_front, key=lambda p: p.mean_score)
                    bd_size = best_dice_p.size_cost(metric)
                    if bd_size != float('inf'):
                        w.add_scalar("pareto/best_dice_size", bd_size, gen)
                self._render_pareto_front(gen, pareto_front, metric)

    def _render_pareto_front(self, gen: int, pareto_front: list,
                             metric: str) -> None:
        """Render the (size, Dice) Pareto front as a scatter+line figure.

        Caller must hold ``self._lock``. Degrades to a no-op when matplotlib is
        unavailable so the GA never crashes on a missing optional dependency.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        pts = [(p.size_cost(metric), p.mean_score) for p in pareto_front]
        pts = [(s, d) for s, d in pts if s != float('inf')]
        if not pts:
            return
        pts.sort(key=lambda t: t[0])
        xs = [s for s, _ in pts]
        ys = [d for _, d in pts]

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.plot(xs, ys, "-o", color="tab:blue")
        ax.set_xlabel(f"model size ({metric})")
        ax.set_ylabel("Dice")
        ax.set_title(f"Pareto front — generation {gen} ({len(pts)} models)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._writer.add_figure("pareto/front", fig, gen)
        plt.close(fig)

    def log_individual(
        self,
        genome: dict,
        mean_score: float,
        std_score: float,
        n_images: int,
        generation: int,
    ) -> None:
        """Log one individual evaluation result.

        Uses a global eval counter as the x-axis so data appears in TensorBoard
        as soon as the first individual completes, without waiting for a full
        generation to finish.
        """
        with self._lock:
            step = self._eval_count
            self._eval_count += 1
            w = self._writer
            # Live per-evaluation stream (one step per trained model). The full
            # per-eval record (std, generation, outcome, …) lives in eval_log.csv.
            w.add_scalar("eval/mean_score", mean_score, step)
            w.add_scalar("eval/n_images",   n_images,   step)
            w.flush()

    def _render_gene_heatmaps(self, gen: int) -> None:
        """Render one convergence heatmap per gene and log it via add_figure.

        Caller must hold ``self._lock``. Each heatmap shows generation (x) vs
        gene value (y); each column is normalised to the population fraction so
        a gene converging collapses to a single bright row. Annotated with the
        fraction when the grid is small enough to stay legible.

        If matplotlib is unavailable in the runtime, degrades gracefully to TB's
        native histograms (numeric genes only) so the GA never crashes on a
        missing optional dependency.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            if not getattr(self, "_warned_no_mpl", False):
                self._warned_no_mpl = True
                print("[TBLogger] matplotlib unavailable — falling back to "
                      "native histograms for genome_dist/*")
            counts = self._gene_hist  # numeric-only fallback
            for gene, history in counts.items():
                vals = []
                for v, c in history.get(gen, {}).items():
                    try:
                        vals.extend([float(v)] * c)
                    except ValueError:
                        pass  # categorical gene: skip in histogram fallback
                if vals:
                    self._writer.add_histogram(
                        f"genome_dist/{gene}", np.asarray(vals), gen)
            return

        for gene, history in self._gene_hist.items():
            gens = sorted(history.keys())
            values = sorted(
                {v for counts in history.values() for v in counts},
                key=_gene_value_sort_key,
            )
            if not values:
                continue

            mat = np.zeros((len(values), len(gens)), dtype=np.float32)
            for j, g in enumerate(gens):
                counts = history[g]
                total = sum(counts.values()) or 1
                for i, v in enumerate(values):
                    mat[i, j] = counts.get(v, 0) / total

            fig_w = max(4.0, 0.5 * len(gens) + 1.5)
            fig_h = max(2.0, 0.45 * len(values) + 1.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(
                mat, aspect="auto", origin="lower",
                cmap="viridis", vmin=0.0, vmax=1.0,
            )
            ax.set_xticks(range(len(gens)))
            ax.set_xticklabels(gens)
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(values)
            ax.set_xlabel("generation")
            ax.set_ylabel(gene)
            ax.set_title(f"{gene} — population fraction")

            # Annotate cells when the grid is small enough to stay readable.
            if len(values) * len(gens) <= 200:
                for i in range(len(values)):
                    for j in range(len(gens)):
                        frac = mat[i, j]
                        if frac > 0:
                            ax.text(
                                j, i, f"{frac:.2f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if frac < 0.6 else "black",
                            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            self._writer.add_figure(f"genome_dist/{gene}", fig, gen)
            plt.close(fig)

    def log_marker_refinement(
        self,
        img_name: str,
        images_folder: str,
        before_seeds: List[Tuple[int, int, int]],
        after_seeds: List[Tuple[int, int, int]],
    ) -> None:
        """Log a before/after side-by-side image showing marker changes.

        Logged under ``marker_refinement/<img_name>`` with an auto-incrementing
        step so every refinement event is preserved in TensorBoard's image
        timeline.
        """
        from PIL import Image as _PIL

        img_path = Path(images_folder.rstrip('/')) / f"{img_name}.png"
        if not img_path.exists():
            return
        try:
            pil = _PIL.open(img_path).convert("RGB")
            orig_W, orig_H = pil.size
            vis_pil = pil.resize((_VIS_SIZE, _VIS_SIZE), _PIL.BILINEAR)
            base = np.array(vis_pil, dtype=np.uint8)

            sx = _VIS_SIZE / orig_W
            sy = _VIS_SIZE / orig_H

            def _scale(seeds):
                return list({(int(x * sx), int(y * sy), lbl) for x, y, lbl in seeds})

            before_vis = _draw_markers_screen(base, _scale(before_seeds), radius=0)
            after_vis  = _draw_markers_screen(base, _scale(after_seeds), radius=0)

            # 4-pixel white separator between before and after panels.
            sep = np.full((_VIS_SIZE, 4, 3), 255, dtype=np.uint8)
            combined = np.concatenate([before_vis, sep, after_vis], axis=1)
            chw = (combined.astype(np.float32) / 255.0).transpose(2, 0, 1)

            with self._lock:
                step = self._refinement_count
                self._refinement_count += 1
                self._writer.add_image(f"marker_refinement/{img_name}", chw, step)
                self._writer.flush()
        except Exception:
            pass

    def close(self) -> None:
        self._writer.close()
