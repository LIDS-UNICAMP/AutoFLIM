"""
GA loop for FLIM encoder NAS.

Each generation:
  1. Select parents with UCB (mean + beta*std) to balance exploitation/exploration.
  2. Produce offspring via uniform crossover + directed/random mutation.
  3. Refine each offspring's training image set via warm-start greedy search.
  4. Keep elite individuals unchanged.
  5. Check convergence and inject diversity on stagnation.

Key design notes:
- Each Individual carries both an architecture genome and a greedily-selected
  training image set; offspring warm-start from their parent's image set.
- ind_cache: evaluated individuals are cached per (genome, image_set) key so
  elite individuals carried across generations are never re-evaluated.
- val_images: an explicit list of images held out as the val set. Greedy
  search and UCB fitness run on train_pool only; val is used solely for
  fitness evaluation after the GA finishes.
- Diversity injection: on population stagnation, the bottom quarter is replaced
  with fresh random individuals instead of stopping immediately.
"""

import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .logging import EvalLogger, TBLogger

import numpy as np

from .genome import SearchSpace, is_valid_genome, random_genome, canonicalize_genome
from .population import (
    Individual,
    adapt_mutation_rate,
    crossover_individuals,
    crowded_compare,
    get_elite_individuals,
    nsga2_select_next_generation,
    nsga2_tournament_select,
    pareto_front_individuals,
    random_mutate,
    ucb_select_individual,
)
from .greedy import dynamic_greedy_refine, cold_start_image, select_diverse_seeds
from .checkpoint import save_checkpoint, capture_rng
from ..arch_space import genome_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per-individual evaluation is retried on failure (typically a transient CUDA
# OOM from concurrent evaluations sharing the GPU). Between attempts the GPU is
# freed and the worker backs off so the concurrent load drains before retrying.
_EVAL_MAX_ATTEMPTS = 5
_EVAL_RETRY_BACKOFF = 5.0   # seconds, multiplied by the attempt number

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GAResult:
    """Result returned by run_ga."""
    generation:      int
    population:      List[Individual]
    best_individual: Individual
    val_images:      List[str]
    train_pool:      List[str]
    # Non-empty only in size_objective='nsga2' mode: the final Pareto front of
    # (Dice, size) trade-offs, best Dice first. best_individual is its max-Dice
    # member in that mode.
    pareto_front:    Optional[List[Individual]] = None

    @property
    def best_genome(self) -> dict:
        return self.best_individual.genome

    @property
    def best_mean(self) -> float:
        return self.best_individual.mean_score

    @property
    def best_std(self) -> float:
        return self.best_individual.std_score


# ---------------------------------------------------------------------------
# Per-individual evaluation
# ---------------------------------------------------------------------------

def _evaluate_individual(
    ind: Individual,
    candidate_pool: List[str],
    dataset_path: Path,
    train_dataset,
    gpu_id: int,
    arch_dir: Optional[Path],
    n_steps: int,
    eps: float,
    selection_strategy: str,
    patience: int,
    ind_cache: Dict[str, Tuple[List[str], float, float, dict]],
    ind_cache_lock: threading.Lock,
    area_prior: Tuple = (None, None),
    position_cache: Optional[dict] = None,
    eval_logger: Optional['EvalLogger'] = None,
    generation: int = 0,
    post_processing: str = 'area_filter',
    color_space_str: str = 'lab',
    use_filter_analysis: bool = True,
    densify_patience: int = 3,
    n_fg_markers_init: int = 20,
    n_bg_markers_init: int = 20,
    max_densify_steps: int = 3,
    densify_threshold: float = 0.3,
    densify_eps: float = 0.02,
    use_inference_guided_markers: bool = False,
    densify_base_count: int = 5,
    densify_scale: float = 3.0,
    densify_mode: str = 'error_region',
    marker_snapshot_root: Optional[Path] = None,
    tb_marker_logger: Optional['TBLogger'] = None,
    max_score_candidates: float = -1,
    score_images=None,
    empty_images=None,
    empty_penalty: float = 0.0,
    stats: Optional[Dict[str, int]] = None,
) -> Individual:
    """
    Run dynamic_greedy_refine on ind and return an updated Individual.
    Results are cached by (genome, image_set) key so elite individuals
    carried unchanged across generations are never re-evaluated.
    The lock protects the shared ind_cache dict for concurrent callers.
    """
    key = ind.cache_key()
    with ind_cache_lock:
        if key in ind_cache:
            cached_imgs, m, s, scores, gains, eff = ind_cache[key]
            ind.image_set        = list(cached_imgs)
            ind.mean_score       = m
            ind.std_score        = s
            ind.per_image_scores = dict(scores)
            ind.image_gains      = dict(gains)
            ind.flops, ind.n_params = eff
            ind.n_evals         += 1
            if stats is not None:
                stats['evals'] = stats.get('evals', 0) + 1
                stats['hits']  = stats.get('hits', 0) + 1
            print(f"  [GA GPU{gpu_id}] cache hit  genome={genome_label(ind.genome)}"
                  f"  mean={m:.4f}  n_imgs={len(cached_imgs)}")
            return ind

    remaining = [img for img in candidate_pool if img not in set(ind.image_set)]

    def _run_refine():
        return dynamic_greedy_refine(
            genome=ind.genome,
            current_images=ind.image_set,
            candidate_pool=remaining,
            dataset_path=dataset_path,
            val_dataset=train_dataset,
            gpu_id=gpu_id,
            arch_dir=arch_dir,
            n_fg_init=n_fg_markers_init,
            n_bg_init=n_bg_markers_init,
            n_steps=n_steps,
            eps=eps,
            patience=patience,
            densify_patience=densify_patience,
            max_densify_steps=max_densify_steps,
            densify_threshold=densify_threshold,
            densify_eps=densify_eps,
            densify_base_count=densify_base_count,
            densify_scale=densify_scale,
            densify_mode=densify_mode,
            use_inference_guided=use_inference_guided_markers,
            area_prior=area_prior,
            position_cache=position_cache,
            eval_logger=eval_logger,
            generation=generation,
            post_processing=post_processing,
            color_space_str=color_space_str,
            erode_radius=float(ind.genome.get('erode_radius', 8.0)),
            selection_strategy=selection_strategy,
            marker_snapshot_root=marker_snapshot_root,
            tb_logger=tb_marker_logger,
            max_score_candidates=max_score_candidates,
            score_images=score_images,
            empty_images=empty_images,
            empty_penalty=empty_penalty,
            use_filter_analysis=use_filter_analysis,
        )

    # Evaluation failures here are almost always transient: a CUDA OOM caused by
    # several individuals being trained/scored on the same GPU at the same time.
    # The fix is not to discard the individual but to retry it after freeing GPU
    # memory and backing off — by then the concurrent evaluations have usually
    # finished and the retry succeeds. A failure is only fatal to the run if it
    # keeps failing across every attempt.
    import time
    import traceback
    last_exc = None
    for attempt in range(1, _EVAL_MAX_ATTEMPTS + 1):
        try:
            (selected, mean_score, std_score, per_image_scores,
             image_gains, efficiency) = _run_refine()
            break
        except Exception as exc:
            last_exc = exc
            # Free whatever this attempt left on the GPU so the retry (and the
            # other concurrent workers) get the memory back.
            try:
                from ..utils import cleanup_cuda_memory
                cleanup_cuda_memory(f"cuda:{gpu_id}", hard=True)
            except Exception:
                pass
            if attempt < _EVAL_MAX_ATTEMPTS:
                backoff = _EVAL_RETRY_BACKOFF * attempt + random.uniform(0, 1.0)
                print(f"  [GA GPU{gpu_id}] eval failed (attempt {attempt}/"
                      f"{_EVAL_MAX_ATTEMPTS}, {type(exc).__name__}: {exc}) — "
                      f"freeing GPU, retrying in {backoff:.1f}s  "
                      f"genome={genome_label(ind.genome)}")
                time.sleep(backoff)
            else:
                print(f"  [GA GPU{gpu_id}] eval still failing after "
                      f"{_EVAL_MAX_ATTEMPTS} attempts  genome={genome_label(ind.genome)}")
                traceback.print_exc()
    else:
        # Exhausted all retries — re-raise so the operator sees a genuinely
        # broken individual rather than a silently mis-scored one.
        raise last_exc

    ind.image_set        = selected
    ind.mean_score       = mean_score
    ind.std_score        = std_score
    ind.n_evals         += 1
    ind.per_image_scores = per_image_scores
    ind.image_gains      = image_gains
    ind.flops, ind.n_params = efficiency

    with ind_cache_lock:
        if key not in ind_cache:
            ind_cache[key] = (
                list(selected), mean_score, std_score,
                dict(per_image_scores), dict(image_gains), efficiency,
            )
        if stats is not None:
            stats['evals'] = stats.get('evals', 0) + 1
    return ind


# ---------------------------------------------------------------------------
# Parallel evaluation batch
# ---------------------------------------------------------------------------

def _eval_population_parallel(
    inds: List[Individual],
    train_pool: List[str],
    dataset_path: Path,
    train_dataset,
    gpu_ids: List[int],
    arch_dir: Optional[Path],
    n_steps: int,
    eps: float,
    selection_strategy: str,
    patience: int,
    ind_cache: Dict,
    ind_cache_lock: threading.Lock,
    area_prior: Tuple,
    position_cache: Dict,
    eval_logger,
    n_parallel: int,
    post_processing: str,
    generation: int,
    color_space_str: str = 'lab',
    use_filter_analysis: bool = True,
    densify_patience: int = 3,
    n_fg_markers_init: int = 20,
    n_bg_markers_init: int = 20,
    max_densify_steps: int = 3,
    densify_threshold: float = 0.3,
    densify_eps: float = 0.02,
    use_inference_guided_markers: bool = False,
    densify_base_count: int = 5,
    densify_scale: float = 3.0,
    densify_mode: str = 'error_region',
    marker_snapshot_root: Optional[Path] = None,
    tb_logger: Optional['TBLogger'] = None,
    tb_marker_logger: Optional['TBLogger'] = None,
    max_score_candidates: float = -1,
    score_images=None,
    empty_images=None,
    empty_penalty: float = 0.0,
    stats: Optional[Dict[str, int]] = None,
) -> None:
    """Evaluate a batch of individuals using a dynamic GPU queue."""
    _adaptive_kwargs = dict(
        densify_patience=densify_patience,
        n_fg_markers_init=n_fg_markers_init,
        n_bg_markers_init=n_bg_markers_init,
        max_densify_steps=max_densify_steps,
        densify_threshold=densify_threshold,
        densify_eps=densify_eps,
        use_inference_guided_markers=use_inference_guided_markers,
        densify_base_count=densify_base_count,
        densify_scale=densify_scale,
        densify_mode=densify_mode,
        marker_snapshot_root=marker_snapshot_root,
        empty_images=empty_images,
        empty_penalty=empty_penalty,
    )

    def _tb_log_ind(ind: Individual) -> None:
        if tb_logger is not None:
            tb_logger.log_individual(
                ind.genome, ind.mean_score, ind.std_score,
                len(ind.image_set), generation,
            )

    if n_parallel == 1:
        for i, ind in enumerate(inds):
            print(f"  Individual {i + 1}/{len(inds)}  genome={genome_label(ind.genome)}")
            _evaluate_individual(
                ind, train_pool, dataset_path,
                train_dataset, gpu_ids[0], arch_dir,
                n_steps, eps, selection_strategy, patience,
                ind_cache, ind_cache_lock, area_prior,
                position_cache=position_cache,
                eval_logger=eval_logger,
                generation=generation,
                post_processing=post_processing,
                color_space_str=color_space_str,
                use_filter_analysis=use_filter_analysis,
                tb_marker_logger=tb_marker_logger,
                max_score_candidates=max_score_candidates,
                score_images=score_images,
                stats=stats,
                **_adaptive_kwargs,
            )
            print(f"    → mean={ind.mean_score:.4f}  n_imgs={len(ind.image_set)}")
            _tb_log_ind(ind)
    else:
        from queue import Queue as _Queue
        gpu_queue: _Queue = _Queue()
        for i in range(n_parallel):
            gpu_queue.put(gpu_ids[i % len(gpu_ids)])

        def _eval_with_gpu(ind):
            g = gpu_queue.get()
            try:
                return _evaluate_individual(
                    ind, train_pool, dataset_path,
                    train_dataset, g, arch_dir,
                    n_steps, eps, selection_strategy, patience,
                    ind_cache, ind_cache_lock, area_prior,
                    position_cache=position_cache,
                    eval_logger=eval_logger,
                    generation=generation,
                    post_processing=post_processing,
                    color_space_str=color_space_str,
                    use_filter_analysis=use_filter_analysis,
                    tb_marker_logger=tb_marker_logger,
                    max_score_candidates=max_score_candidates,
                    score_images=score_images,
                    stats=stats,
                    **_adaptive_kwargs,
                )
            finally:
                gpu_queue.put(g)

        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            fut_to_ind = {pool.submit(_eval_with_gpu, ind): (i, ind)
                          for i, ind in enumerate(inds)}
            for fut in as_completed(fut_to_ind):
                i, ind = fut_to_ind[fut]
                fut.result()
                print(f"  [done {i + 1}/{len(inds)}] genome={genome_label(ind.genome)}"
                      f"  mean={ind.mean_score:.4f}  n_imgs={len(ind.image_set)}")
                _tb_log_ind(ind)


# ---------------------------------------------------------------------------
# Generational loop
# ---------------------------------------------------------------------------

def _evolve(
    population: List[Individual],
    space: SearchSpace,
    train_pool: List[str],
    dataset_path: Path,
    arch_dir: Optional[Path],
    n_generations: int,
    tournament_k: int,
    elite_k: int,
    mutation_rate: float,
    ucb_beta: float,
    n_steps_init: int,
    n_steps_offspring: int,
    eps: float,
    selection_strategy: str,
    patience: int,
    ind_cache: Dict,
    ind_cache_lock: threading.Lock,
    area_prior: Tuple,
    position_cache: Dict,
    eval_logger,
    train_dataset,
    gpu_ids: List[int],
    n_parallel: int,
    no_image_inheritance: bool,
    post_processing: str,
    output_folder: Optional[Path],
    best_ind_init: Individual,
    coreset_image: Optional[str] = None,
    color_space_str: str = 'lab',
    use_filter_analysis: bool = True,
    n_fg_markers_init: int = 20,
    n_bg_markers_init: int = 20,
    max_densify_steps: int = 3,
    densify_threshold: float = 0.3,
    densify_eps: float = 0.02,
    use_inference_guided_markers: bool = False,
    densify_base_count: int = 5,
    densify_scale: float = 3.0,
    densify_mode: str = 'error_region',
    densify_patience: int = 3,
    min_mutation_rate: float = 0.25,
    max_mutation_rate: float = 0.3,
    best_stagnation_patience: int = 0,
    marker_snapshot_root: Optional[Path] = None,
    tb_logger: Optional['TBLogger'] = None,
    max_score_candidates: float = -1,
    score_images=None,
    empty_images=None,
    empty_penalty: float = 0.0,
    reeval_elites: bool = True,
    tb_marker_log_every: int = 1,
    mutation_schedule: str = 'adaptive',
    checkpoint_path: Optional[Path] = None,
    ckpt_config: Optional[dict] = None,
    start_gen: int = 1,
    resume_state: Optional[dict] = None,
) -> Tuple[List[Individual], Individual]:
    """Run n_generations of GA evolution on the population.

    best_ind_init is the best individual seen before this call; it is updated
    and returned as the cumulative best.

    When checkpoint_path is set, the full search state is written once per
    generation so a later --resume continues from start_gen with resume_state's
    counters/RNG restored. ind_cache is persisted so cached evaluations survive.

    Returns (final_population, cumulative_best_individual).
    """
    _pfx = "[GA] "

    _adaptive_kw = dict(
        densify_patience=densify_patience,
        n_fg_markers_init=n_fg_markers_init,
        n_bg_markers_init=n_bg_markers_init,
        max_densify_steps=max_densify_steps,
        densify_threshold=densify_threshold,
        densify_eps=densify_eps,
        use_inference_guided_markers=use_inference_guided_markers,
        densify_base_count=densify_base_count,
        densify_scale=densify_scale,
        densify_mode=densify_mode,
        marker_snapshot_root=marker_snapshot_root,
        empty_images=empty_images,
        empty_penalty=empty_penalty,
    )

    def _eval_parallel(inds: List[Individual], n_steps: int, generation: int,
                       tb_markers=None, stats: Optional[Dict[str, int]] = None) -> None:
        _eval_population_parallel(
            inds, train_pool, dataset_path, train_dataset,
            gpu_ids, arch_dir, n_steps, eps, selection_strategy, patience,
            ind_cache, ind_cache_lock, area_prior, position_cache, eval_logger,
            n_parallel, post_processing, generation=generation,
            color_space_str=color_space_str,
            use_filter_analysis=use_filter_analysis,
            tb_logger=tb_logger,
            tb_marker_logger=tb_markers,
            max_score_candidates=max_score_candidates,
            score_images=score_images,
            stats=stats,
            **_adaptive_kw,
        )

    best_ind = best_ind_init
    stagnation_count = 0
    MAX_STAGNATION   = max(3, n_generations // 3)
    _fresh_ids: set  = set()
    pop_size         = len(population)
    _mutation_rate   = mutation_rate
    # Persistent stagnation-boost multiplier applied on top of the per-schedule
    # base rate (static/linear). For 'adaptive' the boost is applied directly to
    # _mutation_rate (legacy behaviour), so this stays at 1.0 there.
    _boost_mult      = 1.0
    _best_stagnation = 0
    _run_t0          = time.perf_counter()

    # Restore per-mode counters when resuming so the continuation behaves as if
    # the run never stopped (population/best/RNG are restored by the caller).
    # Offset _run_t0 by the pre-crash elapsed time so cumulative wall-time logs
    # stay continuous across the resume.
    if resume_state is not None:
        stagnation_count = resume_state.get('stagnation_count', 0)
        _mutation_rate   = resume_state.get('_mutation_rate', _mutation_rate)
        _boost_mult      = resume_state.get('_boost_mult', 1.0)
        _best_stagnation = resume_state.get('_best_stagnation', 0)
        _run_t0         -= resume_state.get('cumulative_wall_time_s', 0.0)

    def _write_ckpt(gen: int, done: bool) -> None:
        if checkpoint_path is None:
            return
        save_checkpoint(checkpoint_path, {
            'version': 1, 'mode': 'scalar', 'done': done, 'generation': gen,
            'config': ckpt_config or {},
            'population': population, 'best_ind': best_ind,
            'pareto_front': None,
            'ind_cache': dict(ind_cache),
            'rng': capture_rng(),
            'stagnation_count': stagnation_count,
            '_mutation_rate': _mutation_rate,
            '_boost_mult': _boost_mult,
            '_best_stagnation': _best_stagnation,
            'cumulative_wall_time_s': time.perf_counter() - _run_t0,
            'tb_state': tb_logger.export_state() if tb_logger is not None else None,
        })

    for gen in range(start_gen, n_generations + 1):
        _gen_t0 = time.perf_counter()
        # Per-generation diagnostics fed to TensorBoard at the end of the loop.
        gen_stats: Dict[str, int] = {'evals': 0, 'hits': 0}
        n_injected_this_gen = 0
        n_elite_reverted    = 0
        _mut_boosted        = False

        # Set the base mutation rate for this generation according to the chosen
        # schedule. 'adaptive' leaves _mutation_rate as carried from the previous
        # generation's 1/5-rule update; the other two recompute it here so they
        # are not perturbed by adaptation. The stagnation boost (_boost_mult) and
        # the max_mutation_rate ceiling apply in every mode.
        if mutation_schedule == 'static':
            _mutation_rate = min(max_mutation_rate, mutation_rate * _boost_mult)
        elif mutation_schedule == 'linear':
            _frac = (gen - 1) / max(n_generations - 1, 1)
            _base = mutation_rate + (min_mutation_rate - mutation_rate) * _frac
            _mutation_rate = min(max_mutation_rate, _base * _boost_mult)

        effective_beta = ucb_beta * (
            1.0 - 0.5 * (gen - 1) / max(n_generations - 1, 1)
        )
        print(f"\n{_pfx}Generation {gen}/{n_generations}  "
              f"ucb_beta={effective_beta:.3f}")

        # Compute prev_gen_mean over non-elite only so the mutation rate
        # adaptation baseline is not inflated by the carried elite's score.
        _elite_ids = {id(e) for e in get_elite_individuals(population, k=min(elite_k, len(population)))}
        _non_elite = [ind for ind in population if id(ind) not in _elite_ids]
        prev_gen_mean = float(np.mean([ind.mean_score for ind in _non_elite])) if _non_elite else 0.0
        new_population: List[Individual] = []

        elite = get_elite_individuals(population, k=min(elite_k, len(population)))
        new_population.extend(elite)

        existing_genome_keys: set = {
            json.dumps(e.genome, sort_keys=True) for e in elite
        }

        attempts = 0
        # See the NSGA-II loop: a collapsed space cannot supply pop_size unique
        # genomes, so uniqueness has to degrade to a preference eventually.
        _dupes_ok_after = pop_size * 200
        while len(new_population) < pop_size:
            attempts += 1
            _require_unique = attempts <= _dupes_ok_after
            if pop_size * 10 < attempts <= _dupes_ok_after:
                g = random_genome(space)
                g_key = json.dumps(g, sort_keys=True)
                if g_key in existing_genome_keys:
                    continue
                seed = coreset_image if coreset_image is not None else random.choice(train_pool)
                existing_genome_keys.add(g_key)
                new_population.append(Individual(genome=g, image_set=[seed]))
                continue

            parent_a = ucb_select_individual(population, tournament_k, effective_beta)
            parent_b = ucb_select_individual(population, tournament_k, effective_beta)

            if parent_a.genome == parent_b.genome:
                parent_b = ucb_select_individual(population, tournament_k, effective_beta)

            child_genome = crossover_individuals(parent_a, parent_b)
            child_genome = random_mutate(child_genome, space, _mutation_rate)

            if not is_valid_genome(child_genome, space):
                continue

            child_genome = canonicalize_genome(child_genome)

            child_key = json.dumps(child_genome, sort_keys=True)
            if child_key in existing_genome_keys and _require_unique:
                continue
            existing_genome_keys.add(child_key)

            if no_image_inheritance:
                child_images = [coreset_image if coreset_image is not None
                                else random.choice(train_pool)]
            else:
                better_parent = (parent_a if parent_a.mean_score >= parent_b.mean_score
                                 else parent_b)
                child_images = list(better_parent.image_set)

            child = Individual(genome=child_genome, image_set=child_images)
            new_population.append(child)
            if no_image_inheritance:
                _fresh_ids.add(id(child))

        population = new_population

        elite_set = set(id(e) for e in elite)
        for e in elite:
            print(f"  [elite] {genome_label(e.genome)}  mean={e.mean_score:.4f}  (carried)")

        new_inds   = [ind for ind in population if id(ind) not in elite_set]
        fresh_inds = [ind for ind in new_inds if id(ind) in _fresh_ids]
        warm_inds  = [ind for ind in new_inds if id(ind) not in _fresh_ids]
        _fresh_ids.clear()

        if fresh_inds:
            print(f"{_pfx}  Evaluating {len(fresh_inds)} injected individuals "
                  f"(n_steps_init={n_steps_init})")
            _eval_parallel(fresh_inds, n_steps_init, generation=gen, stats=gen_stats)
        if warm_inds:
            print(f"{_pfx}  Evaluating {len(warm_inds)} offspring "
                  f"(n_steps_offspring={n_steps_offspring})")
            _eval_parallel(warm_inds, n_steps_offspring, generation=gen, stats=gen_stats)

        # Re-evaluate elite every 2 generations so their training sets can grow
        # and stale cached scores don't block the population from improving.
        # If the new score is lower (unlucky markers), revert BOTH mean_score AND
        # image_set together to keep the individual in a consistent state.
        if reeval_elites and gen % 2 == 0 and elite:
            _state_before = {
                id(e): (e.mean_score, e.std_score, list(e.image_set),
                        dict(e.per_image_scores), dict(e.image_gains))
                for e in elite
            }
            # Back up each elite's marker snapshot before re-evaluation. The
            # re-eval regenerates fresh random markers and overwrites the
            # snapshot dir (keyed on genome+image_set) whenever it ends on the
            # same image set. If the new draw scores worse we revert the score
            # below, but WITHOUT restoring the snapshot the final model would
            # later be retrained from the new (worse) markers — so the logged
            # best score and the shipped model disagree. This is harmless for
            # draw-robust learners (cluster) but collapses draw-sensitive ones
            # (bofp): see exp_ga_dynamic_100pct_3 (search 0.8333 vs final 0.3722).
            _snap_backup: Dict[int, Tuple[Path, Path]] = {}
            if marker_snapshot_root is not None:
                from ..arch_space import marker_snapshot_subdir
                for e in elite:
                    _old_imgs = _state_before[id(e)][2]
                    if not _old_imgs:
                        continue
                    _snap_dir = (Path(marker_snapshot_root)
                                 / marker_snapshot_subdir(e.genome, _old_imgs))
                    if _snap_dir.exists():
                        _bak = _snap_dir.parent / (_snap_dir.name + ".revert_bak")
                        shutil.rmtree(_bak, ignore_errors=True)
                        shutil.copytree(_snap_dir, _bak)
                        _snap_backup[id(e)] = (_snap_dir, _bak)
            with ind_cache_lock:
                for e in elite:
                    ind_cache.pop(e.cache_key(), None)
            print(f"{_pfx}  Re-evaluating {len(elite)} elite "
                  f"(n_steps_offspring={n_steps_offspring})")
            # Marker images are logged only for the generation-best individual
            # (dedicated pass below), so elite re-eval does not pass tb_markers.
            _eval_parallel(elite, n_steps_offspring, generation=gen,
                           stats=gen_stats)
            for e in elite:
                old_score, old_std, old_imgs, old_per_img, old_gains = _state_before[id(e)]
                if e.mean_score < old_score:
                    n_elite_reverted  += 1
                    e.mean_score       = old_score
                    e.std_score        = old_std
                    e.image_set        = old_imgs
                    e.per_image_scores = old_per_img
                    e.image_gains      = old_gains
                    # Restore the reverted state in the cache so offspring that
                    # inherit this image_set get a cache hit instead of being
                    # re-evaluated with fresh random markers.
                    with ind_cache_lock:
                        ind_cache[e.cache_key()] = (
                            list(old_imgs), old_score, old_std,
                            dict(old_per_img), dict(old_gains),
                            (e.flops, e.n_params),
                        )
                    # Roll the on-disk markers back to the draw that produced
                    # old_score so the final model is retrained from the same
                    # markers the reverted score reflects.
                    _bak_entry = _snap_backup.get(id(e))
                    if _bak_entry is not None:
                        _snap_dir, _bak = _bak_entry
                        shutil.rmtree(_snap_dir, ignore_errors=True)
                        shutil.copytree(_bak, _snap_dir)
            # Drop the backups (kept snapshots stay; reverted ones already restored).
            for _snap_dir, _bak in _snap_backup.values():
                shutil.rmtree(_bak, ignore_errors=True)

        if new_inds and mutation_schedule == 'adaptive':
            n_improved = sum(1 for ind in new_inds if ind.mean_score > prev_gen_mean)
            _mutation_rate = adapt_mutation_rate(_mutation_rate, n_improved, len(new_inds),
                                                  min_rate=min_mutation_rate,
                                                  max_rate=max_mutation_rate)

        current_best = max(population, key=lambda x: x.mean_score)
        if current_best.mean_score > best_ind.mean_score:
            best_ind = current_best
            _best_stagnation = 0
            _boost_mult = 1.0
        else:
            _best_stagnation += 1
            if _best_stagnation % 4 == 0:
                # Gentle nudge capped at the max_mutation_rate ceiling; a per-gene
                # rate above that flips most of the genome and destroys crossover
                # heritability (degenerates to random search).
                if mutation_schedule == 'adaptive':
                    _mutation_rate = min(max_mutation_rate, _mutation_rate * 1.1)
                else:
                    # Persist the boost as a multiplier; the per-schedule base
                    # rate at the top of the loop re-applies it (and the ceiling).
                    _boost_mult *= 1.1
                    _mutation_rate = min(max_mutation_rate, _mutation_rate * 1.1)
                _mut_boosted = True
                print(f"{_pfx}Best stagnated {_best_stagnation} gens — "
                      f"boosting mut_rate → {_mutation_rate:.3f}")
        print(f"{_pfx}Gen {gen} best: {genome_label(current_best.genome)}  "
              f"mean={current_best.mean_score:.4f}  overall best={best_ind.mean_score:.4f}  "
              f"mut_rate={_mutation_rate:.3f}")

        # --- Marker-refinement logging for the generation-best individual only ---
        # Replays the best individual's adaptive refinement with TensorBoard image
        # logging ON and UNCAPPED (every refined image per densify step, not just
        # tb_max_refinement_images). Runs in an isolated temp dir with no snapshot,
        # cache, or eval-log writes (marker_snapshot_root/eval_logger=None, private
        # position_cache), so the discarded result never perturbs the search.
        if (tb_logger is not None
                and tb_marker_log_every > 0 and gen % tb_marker_log_every == 0):
            try:
                # Replay from a COLD seed (single image) through the full
                # n_steps_init build-up. Densification — and therefore marker-
                # refinement logging — fires on the small/undertrained model
                # early in the build-up; starting from the best's already-
                # converged image set only triggers ADD_IMAGE (low confusion →
                # no densify → no marker images).
                _seed = current_best.image_set[0]
                _best_remaining = [img for img in train_pool if img != _seed]
                print(f"{_pfx}  [tb] logging marker refinement for gen-best "
                      f"({genome_label(current_best.genome)})")
                dynamic_greedy_refine(
                    genome=current_best.genome,
                    current_images=[_seed],
                    candidate_pool=_best_remaining,
                    dataset_path=dataset_path,
                    val_dataset=train_dataset,
                    gpu_id=gpu_ids[0],
                    arch_dir=arch_dir,
                    n_fg_init=n_fg_markers_init,
                    n_bg_init=n_bg_markers_init,
                    n_steps=n_steps_init,
                    eps=eps,
                    patience=patience,
                    densify_patience=densify_patience,
                    max_densify_steps=max_densify_steps,
                    densify_threshold=densify_threshold,
                    densify_eps=densify_eps,
                    densify_base_count=densify_base_count,
                    densify_scale=densify_scale,
                    densify_mode=densify_mode,
                    use_inference_guided=use_inference_guided_markers,
                    area_prior=area_prior,
                    position_cache={},
                    eval_logger=None,
                    generation=gen,
                    post_processing=post_processing,
                    color_space_str=color_space_str,
                    erode_radius=float(current_best.genome.get('erode_radius', 8.0)),
                    selection_strategy=selection_strategy,
                    marker_snapshot_root=None,     # non-destructive: no snapshot writes
                    tb_logger=tb_logger,
                    tb_max_refinement_images=-1,   # uncapped: log every refined image
                    tb_force_densify=True,         # guarantee densify fires so marker
                                                   # images are actually emitted (this
                                                   # replay's result is discarded)
                    max_score_candidates=max_score_candidates,
                    score_images=score_images,
                )
            except Exception as exc:
                print(f"{_pfx}  [tb] gen-best marker logging failed: {exc}")

        means = [ind.mean_score for ind in population]
        if np.std(means) < 0.001:
            stagnation_count += 1
            print(f"{_pfx}Stagnation {stagnation_count}/{MAX_STAGNATION}")
            if stagnation_count >= MAX_STAGNATION:
                print(f"{_pfx}Stopping — population converged.")
                break
            n_inject = max(1, pop_size // 4)
            worst_idx = sorted(range(len(population)),
                               key=lambda i: population[i].mean_score)
            injected: List[Individual] = []
            for j in worst_idx[:n_inject]:
                g = random_genome(space)
                seed = (coreset_image if coreset_image is not None
                        else random.choice(train_pool))
                new_ind = Individual(genome=g, image_set=[seed])
                population[j] = new_ind
                injected.append(new_ind)
            # Evaluate the injected individuals now so they enter the population
            # with real fitness. Otherwise they sit at score 0 with n_evals=0,
            # which gives them the maximum UCB bonus — they dominate parent
            # selection for one generation and are then discarded unscored.
            if injected:
                _eval_parallel(injected, n_steps_init, generation=gen, stats=gen_stats)
                n_injected_this_gen = len(injected)
                inj_best = max(injected, key=lambda x: x.mean_score)
                if inj_best.mean_score > best_ind.mean_score:
                    best_ind = inj_best
            print(f"{_pfx}Injected and evaluated {n_inject} random individuals.")
        else:
            stagnation_count = 0

        if tb_logger is not None:
            _now = time.perf_counter()
            tb_logger.log_generation(
                gen=gen,
                population=population,
                best_ind=best_ind,
                current_best=current_best,
                mutation_rate=_mutation_rate,
                effective_beta=effective_beta,
                best_stagnation=_best_stagnation,
                pop_stagnation=stagnation_count,
                n_injected=n_injected_this_gen,
                n_elite_reverted=n_elite_reverted,
                mutation_boosted=_mut_boosted,
                n_evals=gen_stats['evals'],
                n_cache_hits=gen_stats['hits'],
                gen_wall_time_s=_now - _gen_t0,
                cumulative_wall_time_s=_now - _run_t0,
            )

        if output_folder is not None:
            _log_generation(gen, population, best_ind, output_folder)

        # Optional early stop: end the run once the global best has not improved for
        # best_stagnation_patience generations. Disabled when patience <= 0. Placed
        # after this generation is fully logged so nothing is lost on the way out.
        _early_stop = bool(best_stagnation_patience
                           and _best_stagnation >= best_stagnation_patience)
        # Checkpoint AFTER the generation is fully logged. done=True marks the run
        # complete (last gen or early stop) so --resume goes straight to saving
        # instead of evolving further.
        _write_ckpt(gen, done=_early_stop or gen == n_generations)
        if _early_stop:
            print(f"{_pfx}Stopping early — best stagnated {_best_stagnation} gens "
                  f"(>= best_stagnation_patience={best_stagnation_patience}).")
            break

    return population, best_ind


def _evolve_nsga2(
    population: List[Individual],
    space: SearchSpace,
    train_pool: List[str],
    dataset_path: Path,
    arch_dir: Optional[Path],
    n_generations: int,
    tournament_k: int,
    mutation_rate: float,
    n_steps_init: int,
    n_steps_offspring: int,
    eps: float,
    selection_strategy: str,
    patience: int,
    ind_cache: Dict,
    ind_cache_lock: threading.Lock,
    area_prior: Tuple,
    position_cache: Dict,
    eval_logger,
    train_dataset,
    gpu_ids: List[int],
    n_parallel: int,
    no_image_inheritance: bool,
    post_processing: str,
    output_folder: Optional[Path],
    best_ind_init: Individual,
    efficiency_metric: str = 'flops',
    coreset_image: Optional[str] = None,
    color_space_str: str = 'lab',
    use_filter_analysis: bool = True,
    n_fg_markers_init: int = 20,
    n_bg_markers_init: int = 20,
    max_densify_steps: int = 3,
    densify_threshold: float = 0.3,
    densify_eps: float = 0.02,
    use_inference_guided_markers: bool = False,
    densify_base_count: int = 5,
    densify_scale: float = 3.0,
    densify_mode: str = 'error_region',
    densify_patience: int = 3,
    min_mutation_rate: float = 0.25,
    max_mutation_rate: float = 0.3,
    best_stagnation_patience: int = 0,
    marker_snapshot_root: Optional[Path] = None,
    tb_logger: Optional['TBLogger'] = None,
    max_score_candidates: float = -1,
    score_images=None,
    empty_images=None,
    empty_penalty: float = 0.0,
    mutation_schedule: str = 'static',
    checkpoint_path: Optional[Path] = None,
    ckpt_config: Optional[dict] = None,
    start_gen: int = 1,
    resume_state: Optional[dict] = None,
) -> Tuple[List[Individual], Individual, List[Individual]]:
    """NSGA-II (mu+lambda) multi-objective evolution: maximise Dice, minimise size.

    Replaces the scalar elite/UCB machinery of ``_evolve``: parents are chosen by
    binary tournament under the crowded-comparison operator, and survivors are
    picked front-by-front with crowding-distance truncation. Diversity is handled
    intrinsically by crowding (no UCB exploration bonus, no random injection).

    Returns (final_population, best_individual, pareto_front), where
    best_individual is the max-Dice member of the final non-dominated front.
    """
    _pfx = "[GA-NSGA2] "

    _adaptive_kw = dict(
        densify_patience=densify_patience,
        n_fg_markers_init=n_fg_markers_init,
        n_bg_markers_init=n_bg_markers_init,
        max_densify_steps=max_densify_steps,
        densify_threshold=densify_threshold,
        densify_eps=densify_eps,
        use_inference_guided_markers=use_inference_guided_markers,
        densify_base_count=densify_base_count,
        densify_scale=densify_scale,
        densify_mode=densify_mode,
        marker_snapshot_root=marker_snapshot_root,
        empty_images=empty_images,
        empty_penalty=empty_penalty,
    )

    def _eval_parallel(inds: List[Individual], n_steps: int, generation: int,
                       stats: Optional[Dict[str, int]] = None) -> None:
        _eval_population_parallel(
            inds, train_pool, dataset_path, train_dataset,
            gpu_ids, arch_dir, n_steps, eps, selection_strategy, patience,
            ind_cache, ind_cache_lock, area_prior, position_cache, eval_logger,
            n_parallel, post_processing, generation=generation,
            color_space_str=color_space_str,
            use_filter_analysis=use_filter_analysis,
            tb_logger=tb_logger,
            max_score_candidates=max_score_candidates,
            score_images=score_images,
            stats=stats,
            **_adaptive_kw,
        )

    pop_size = len(population)
    best_ind = best_ind_init
    _run_t0  = time.perf_counter()

    # Seed rank/crowding on the (possibly restored) population so the first
    # tournament is valid. Deterministic from the individuals' scores/sizes.
    population = nsga2_select_next_generation(population, pop_size, efficiency_metric)
    pareto_front = pareto_front_individuals(population, efficiency_metric)
    _best_front_dice = max((p.mean_score for p in pareto_front), default=0.0)
    _best_stagnation = 0
    # On resume, keep the stagnation counters from the checkpoint so the early-
    # stop fires at the same point an uninterrupted run would have, and offset
    # _run_t0 so cumulative wall-time logs stay continuous.
    if resume_state is not None:
        _best_front_dice = resume_state.get('_best_front_dice', _best_front_dice)
        _best_stagnation = resume_state.get('_best_stagnation', 0)
        _run_t0         -= resume_state.get('cumulative_wall_time_s', 0.0)

    def _size_str(ind: Individual) -> str:
        c = ind.size_cost(efficiency_metric)
        return "inf" if c == float('inf') else f"{c:.3e}"

    def _write_ckpt(gen: int, done: bool) -> None:
        if checkpoint_path is None:
            return
        save_checkpoint(checkpoint_path, {
            'version': 1, 'mode': 'nsga2', 'done': done, 'generation': gen,
            'config': ckpt_config or {},
            'population': population, 'best_ind': best_ind,
            'pareto_front': pareto_front,
            'ind_cache': dict(ind_cache) if ind_cache is not None else {},
            'rng': capture_rng(),
            '_best_front_dice': _best_front_dice,
            '_best_stagnation': _best_stagnation,
            'cumulative_wall_time_s': time.perf_counter() - _run_t0,
            'tb_state': tb_logger.export_state() if tb_logger is not None else None,
        })

    for gen in range(start_gen, n_generations + 1):
        _gen_t0 = time.perf_counter()
        gen_stats: Dict[str, int] = {'evals': 0, 'hits': 0}

        # Mutation rate schedule (static/linear; 'adaptive' treated as static
        # since the 1/5-rule baseline has no single-objective meaning here).
        if mutation_schedule == 'linear':
            _frac = (gen - 1) / max(n_generations - 1, 1)
            _mutation_rate = mutation_rate + (min_mutation_rate - mutation_rate) * _frac
        else:
            _mutation_rate = mutation_rate
        _mutation_rate = min(max_mutation_rate, _mutation_rate)

        print(f"\n{_pfx}Generation {gen}/{n_generations}  "
              f"front0={len(pareto_front)}  mut_rate={_mutation_rate:.3f}")

        # ---- Produce offspring via crowded-comparison tournaments ----
        existing_genome_keys: set = {
            json.dumps(ind.genome, sort_keys=True) for ind in population
        }
        offspring: List[Individual] = []
        attempts = 0
        # A space holding fewer distinct genomes than pop_size (a collapsed
        # architecture, e.g. the fixed-arch ablation) admits no unique child, so
        # past this point uniqueness is a preference rather than a requirement.
        _dupes_ok_after = pop_size * 200
        while len(offspring) < pop_size:
            attempts += 1
            _require_unique = attempts <= _dupes_ok_after
            if pop_size * 20 < attempts <= _dupes_ok_after:
                # Search space exhausted around the current front — pad with a
                # fresh random genome so the generation still fills.
                g = canonicalize_genome(random_genome(space))
                g_key = json.dumps(g, sort_keys=True)
                if g_key in existing_genome_keys:
                    continue
                existing_genome_keys.add(g_key)
                seed = coreset_image if coreset_image is not None else random.choice(train_pool)
                offspring.append(Individual(genome=g, image_set=[seed]))
                continue

            parent_a = nsga2_tournament_select(population, tournament_k)
            parent_b = nsga2_tournament_select(population, tournament_k)
            if parent_a.genome == parent_b.genome:
                parent_b = nsga2_tournament_select(population, tournament_k)

            child_genome = crossover_individuals(parent_a, parent_b)
            child_genome = random_mutate(child_genome, space, _mutation_rate)
            if not is_valid_genome(child_genome, space):
                continue
            child_genome = canonicalize_genome(child_genome)
            child_key = json.dumps(child_genome, sort_keys=True)
            if child_key in existing_genome_keys and _require_unique:
                continue
            existing_genome_keys.add(child_key)

            if no_image_inheritance:
                child_images = [coreset_image if coreset_image is not None
                                else random.choice(train_pool)]
            else:
                # Inherit the image set of the parent that wins crowded comparison.
                better_parent = crowded_compare(parent_a, parent_b)
                child_images = list(better_parent.image_set)

            offspring.append(Individual(genome=child_genome, image_set=child_images))

        # ---- Evaluate offspring ----
        # Warm = inherited a multi-image set from a parent (refine from there);
        # cold = single seed image (full build-up). Split by image-set size.
        warm = [o for o in offspring if len(o.image_set) > 1]
        cold = [o for o in offspring if len(o.image_set) <= 1]
        if cold:
            print(f"{_pfx}  Evaluating {len(cold)} cold offspring "
                  f"(n_steps_init={n_steps_init})")
            _eval_parallel(cold, n_steps_init, generation=gen, stats=gen_stats)
        if warm:
            print(f"{_pfx}  Evaluating {len(warm)} warm offspring "
                  f"(n_steps_offspring={n_steps_offspring})")
            _eval_parallel(warm, n_steps_offspring, generation=gen, stats=gen_stats)

        # ---- mu+lambda survivor selection ----
        combined = population + offspring
        population = nsga2_select_next_generation(combined, pop_size, efficiency_metric)
        pareto_front = pareto_front_individuals(population, efficiency_metric)

        # best = max-Dice member of the current Pareto front (cumulative).
        gen_best = pareto_front[0] if pareto_front else max(population, key=lambda x: x.mean_score)
        if gen_best.mean_score > best_ind.mean_score:
            best_ind = gen_best

        front_dice = max((p.mean_score for p in pareto_front), default=0.0)
        if front_dice > _best_front_dice + 1e-6:
            _best_front_dice = front_dice
            _best_stagnation = 0
        else:
            _best_stagnation += 1

        print(f"{_pfx}Gen {gen} front0={len(pareto_front)}  "
              f"best_dice={front_dice:.4f}  "
              f"size[{efficiency_metric}]: "
              f"{_size_str(pareto_front[0]) if pareto_front else 'n/a'} … "
              f"{_size_str(pareto_front[-1]) if pareto_front else 'n/a'}")
        for p in pareto_front:
            print(f"    [front0] {genome_label(p.genome)}  "
                  f"dice={p.mean_score:.4f}  size[{efficiency_metric}]={_size_str(p)}")

        if tb_logger is not None:
            _now = time.perf_counter()
            tb_logger.log_generation(
                gen=gen, population=population,
                best_ind=best_ind, current_best=gen_best,
                mutation_rate=_mutation_rate, effective_beta=0.0,
                best_stagnation=_best_stagnation,
                n_evals=gen_stats['evals'], n_cache_hits=gen_stats['hits'],
                gen_wall_time_s=_now - _gen_t0,
                cumulative_wall_time_s=_now - _run_t0,
                pareto_front=pareto_front,
                efficiency_metric=efficiency_metric,
            )
        if output_folder is not None:
            _log_generation(gen, population, best_ind, output_folder)

        _early_stop = bool(best_stagnation_patience
                           and _best_stagnation >= best_stagnation_patience)
        # Checkpoint after the generation is fully logged; done=True on the last
        # gen or an early stop so --resume goes straight to saving.
        _write_ckpt(gen, done=_early_stop or gen == n_generations)
        if _early_stop:
            print(f"{_pfx}Stopping early — front Dice stagnated {_best_stagnation} gens "
                  f"(>= best_stagnation_patience={best_stagnation_patience}).")
            break

    pareto_front = pareto_front_individuals(population, efficiency_metric)
    return population, best_ind, pareto_front


def _log_generation(gen: int, population: List[Individual],
                    best: Individual, output_folder: Path) -> None:
    """Append a one-row-per-individual CSV log for the GA."""
    import csv
    log_path = output_folder / "ga_log.csv"
    write_header = not log_path.exists()
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['generation', 'genome', 'mean_score', 'std_score',
                             'n_images', 'flops', 'n_params', 'image_set', 'is_best'])
        for ind in population:
            writer.writerow([
                gen,
                genome_label(ind.genome),
                f"{ind.mean_score:.4f}",
                f"{ind.std_score:.4f}",
                len(ind.image_set),
                ind.flops if ind.flops is not None else '',
                ind.n_params if ind.n_params is not None else '',
                ';'.join(ind.image_set),
                int(ind is best),
            ])


# ---------------------------------------------------------------------------
# GA entry point
# ---------------------------------------------------------------------------

def run_ga(
    space: SearchSpace,
    image_names: List[str],
    dataset_path: Path,
    gpu_id: int,
    coreset_image: Optional[str] = None,
    population_size: int = 10,
    n_generations: int = 10,
    n_steps_init: int = 5,
    n_steps_offspring: int = 3,
    eps: float = 0.005,
    selection_strategy: str = 'worst_first',
    patience: int = 3,
    tournament_k: int = 3,
    elite_k: int = 2,
    mutation_rate: float = 0.2,
    ucb_beta: float = 0.5,
    arch_dir: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    initial_population: Optional[List[dict]] = None,
    val_images: Optional[List[str]] = None,
    val_empty_images: Optional[List[str]] = None,
    empty_penalty: float = 0.0,
    color_space_str: str = 'lab',
    cold_start_strategy: str = 'coreset',
    gpu_ids: Optional[List[int]] = None,
    num_workers_evaluation: Optional[int] = None,
    no_image_inheritance: bool = False,
    post_processing: str = 'area_filter',
    use_filter_analysis: bool = True,
    n_fg_markers_init: int = 20,
    n_bg_markers_init: int = 20,
    max_densify_steps: int = 3,
    densify_threshold: float = 0.3,
    densify_eps: float = 0.02,
    use_inference_guided_markers: bool = False,
    densify_base_count: int = 5,
    densify_scale: float = 3.0,
    densify_mode: str = 'error_region',
    densify_patience: int = 3,
    min_mutation_rate: float = 0.25,
    max_mutation_rate: float = 0.3,
    best_stagnation_patience: int = 0,
    marker_snapshot_root: Optional[Path] = None,
    tb_logger: Optional['TBLogger'] = None,
    max_score_candidates: float = -1,
    reeval_elites: bool = True,
    tb_marker_log_every: int = 1,
    mutation_schedule: str = 'adaptive',
    size_objective: str = 'none',
    efficiency_metric: str = 'flops',
    resume: bool = False,
    seed: Optional[int] = None,
) -> GAResult:
    """
    GA: co-evolves architecture genomes and training image sets.

    Generation 0: cold-start image(s) assigned by cold_start_strategy:
      'coreset' — all individuals share the image closest to the dataset mean.
      'random'  — each individual draws an independent random image.
      'diverse' — seeds spread via greedy farthest-point sampling.

    Generation k: crossover + mutation → warm-start greedy refinement
    (offspring inherit the better parent's image set).

    Fitness for UCB selection is mean Dice on the fixed val set after
    training on the individual's image_set.
    """
    from pyflim.datasets.flim_standard import FLIMStandardDataset
    from .checkpoint import (load_checkpoint, restore_rng, config_fingerprint,
                             config_matches)

    _tag = "[GA]"

    # ---- Reproducibility ----
    # Seed the global RNGs before any random draw (cold-start seeds, genome
    # sampling) so a fresh run is reproducible and a --resume continuation is
    # meaningful. The checkpoint also captures/restores RNG state per generation,
    # so resume continues the exact same stream regardless of this seed.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"{_tag} RNG seeded with {seed}")

    # ---- Fixed val/train split ----
    # train_pool = the candidate/training images (individuals select and train
    # from here). GA fitness is scored on the val_csv images (val_images), which
    # are disjoint from train_pool so they are never trained on.
    # NOTE: val_csv is the GA's model-selection signal, so the final-block test
    # on the same val_csv is an optimistic (not strictly held-out) estimate.
    # Empty-penalty mode: empty-GT val images are pulled out of the mean-Dice
    # fitness and instead contribute a false-positive penalty (_fitness_stats).
    # Off (empty_penalty<=0) → empty_images stays empty and scoring is unchanged.
    empty_images = (set(val_empty_images)
                    if (val_empty_images and empty_penalty > 0) else set())
    if val_images is not None and len(val_images) > 0:
        val_set    = set(val_images)
        train_pool = [img for img in image_names if img not in val_set]
        if not train_pool:
            train_pool = image_names
        # Foreground val images drive the mean; empties (if any) are scored too
        # but only feed the penalty term, so keep them out of score_images.
        score_images = set(val_images) - empty_images
    else:
        val_images   = image_names
        train_pool   = image_names
        score_images = None   # no val set → score over the whole train_pool

    print(f"{_tag} val={len(val_images)}  train_pool={len(train_pool)}  "
          f"score_on_val={len(score_images) if score_images else 0}  "
          f"empty_penalty={empty_penalty}  empty_val={len(empty_images)}")

    images_folder = str(dataset_path / "images") + "/"
    labels_folder = str(dataset_path / "truelabels") + "/"

    train_pool_dataset = FLIMStandardDataset(
        image_folder=images_folder,
        images_list=train_pool,
        label_folder=labels_folder,
        orig_ext=".png",
        label_ext="png",
        lab_norm=True,
    )
    from ..utils import compute_area_prior
    area_prior = compute_area_prior(train_pool_dataset)
    print(f"{_tag} area_prior: lo={area_prior[0]:.1f}  hi={area_prior[1]:.1f}"
          if area_prior[0] is not None else f"{_tag} area_prior: no foreground found")

    # Evaluation dataset covers BOTH train_pool (per-image scores drive worst_first
    # candidate ranking) and the val images (fitness), so one pass scores both.
    _scoring_images = sorted(set(train_pool) | set(val_images))
    scoring_dataset = FLIMStandardDataset(
        image_folder=images_folder,
        images_list=_scoring_images,
        label_folder=labels_folder,
        orig_ext=".png",
        label_ext="png",
        lab_norm=True,
    )

    # ---- Cold-start seeds (drawn from train_pool; seeds become training images) ----
    _strategy = cold_start_strategy if coreset_image is None else 'coreset'
    if _strategy == 'coreset':
        if coreset_image is None:
            coreset_image = cold_start_image(train_pool, images_folder, color_space_str)
        print(f"{_tag} cold_start=coreset  seed={coreset_image}")
        _seeds = [coreset_image] * population_size
    elif _strategy == 'random':
        _seeds = [random.choice(train_pool) for _ in range(population_size)]
        print(f"{_tag} cold_start=random  seeds={_seeds}")
    elif _strategy == 'diverse':
        _seeds = select_diverse_seeds(
            train_pool, images_folder, n=population_size, color_space_str=color_space_str
        )
        while len(_seeds) < population_size:
            _seeds.append(random.choice(train_pool))
        print(f"{_tag} cold_start=diverse  seeds={_seeds}")
    else:
        raise ValueError(f"Unknown cold_start_strategy={_strategy!r}. "
                         f"Choose 'coreset', 'random', or 'diverse'.")

    # ---- GPU / parallelism ----
    _gpu_ids = gpu_ids if gpu_ids else [gpu_id]
    n_parallel = num_workers_evaluation if num_workers_evaluation else len(_gpu_ids)
    print(f"{_tag} parallel eval workers={n_parallel}  gpu_ids={_gpu_ids}")

    # ---- Shared caches ----
    global_position_cache: Dict = {}
    from .logging import EvalLogger
    eval_logger: Optional[EvalLogger] = None
    if output_folder is not None:
        eval_logger = EvalLogger(output_folder / 'eval_log.csv')
        print(f"{_tag} eval log → {output_folder / 'eval_log.csv'}")

    _adaptive_kw = dict(
        densify_patience=densify_patience,
        n_fg_markers_init=n_fg_markers_init,
        n_bg_markers_init=n_bg_markers_init,
        max_densify_steps=max_densify_steps,
        densify_threshold=densify_threshold,
        densify_eps=densify_eps,
        use_inference_guided_markers=use_inference_guided_markers,
        densify_base_count=densify_base_count,
        densify_scale=densify_scale,
        densify_mode=densify_mode,
        marker_snapshot_root=marker_snapshot_root,
        empty_images=empty_images,
        empty_penalty=empty_penalty,
        #min_mutation_rate=min_mutation_rate,
    )

    # ---- Initial genomes ----
    init_genomes = [canonicalize_genome(dict(g)) for g in initial_population[:population_size]] \
                   if initial_population else []
    while len(init_genomes) < population_size:
        init_genomes.append(random_genome(space))

    ind_cache: Dict = {}
    ind_cache_lock = threading.Lock()

    # ---- Checkpoint / resume setup ----
    checkpoint_path = (output_folder / "ga_checkpoint.pkl"
                       if output_folder is not None else None)
    ckpt_config = config_fingerprint(
        population_size=population_size, n_generations=n_generations,
        size_objective=size_objective, efficiency_metric=efficiency_metric,
        n_train_pool=len(train_pool), n_val=len(val_images),
    )
    resume_state: Optional[dict] = None
    start_gen = 1
    if resume and checkpoint_path is not None:
        _ck = load_checkpoint(checkpoint_path)
        if _ck is not None and config_matches(_ck, ckpt_config):
            restore_rng(_ck.get('rng'))
            ind_cache.update(_ck.get('ind_cache', {}))
            # Restore the heatmap accumulation so genome_dist/* render over the
            # full generation range, not just from the resume point onward.
            if tb_logger is not None:
                tb_logger.import_state(_ck.get('tb_state'))
            if _ck.get('done'):
                # Search already finished before the crash — skip straight to
                # returning the saved result so the caller proceeds to save.
                print(f"{_tag} Resume: checkpoint marks the search DONE at gen "
                      f"{_ck['generation']} — skipping evolution.")
                return GAResult(
                    generation=_ck['generation'],
                    population=_ck['population'],
                    best_individual=_ck['best_ind'],
                    val_images=val_images,
                    train_pool=train_pool,
                    pareto_front=_ck.get('pareto_front'),
                )
            resume_state = _ck
            start_gen = _ck['generation'] + 1
            print(f"{_tag} Resume: continuing from generation {start_gen} "
                  f"(checkpoint at gen {_ck['generation']}, "
                  f"{len(ind_cache)} cached evals restored).")

    if resume_state is not None:
        # Restore the evolved population/best instead of cold-starting gen 0.
        population = resume_state['population']
        best_ind = resume_state['best_ind']
        pareto_front: Optional[List[Individual]] = resume_state.get('pareto_front')
        print(f"{_tag} Restored population of {len(population)} individuals; "
              f"best mean={best_ind.mean_score:.4f}")
    else:
        population = [
            Individual(genome=g, image_set=[s])
            for g, s in zip(init_genomes, _seeds)
        ]

        print(f"\n{_tag} Generation 0 — evaluating {len(population)} individuals "
              f"(n_steps_init={n_steps_init})")
        _eval_population_parallel(
            population, train_pool, dataset_path, scoring_dataset,
            _gpu_ids, arch_dir, n_steps_init, eps, selection_strategy, patience,
            ind_cache, ind_cache_lock, area_prior, global_position_cache, eval_logger,
            n_parallel, post_processing, generation=0,
            color_space_str=color_space_str,
            use_filter_analysis=use_filter_analysis,
            tb_logger=tb_logger,
            max_score_candidates=max_score_candidates,
            score_images=score_images,
            **_adaptive_kw,
        )

        best_ind = max(population, key=lambda x: x.mean_score)
        print(f"{_tag} Gen 0 best: {genome_label(best_ind.genome)}  "
              f"mean={best_ind.mean_score:.4f}  n_imgs={len(best_ind.image_set)}")

        if tb_logger is not None:
            tb_logger.log_generation(
                gen=0, population=population,
                best_ind=best_ind, current_best=best_ind,
                mutation_rate=mutation_rate, effective_beta=ucb_beta,
            )
        pareto_front: Optional[List[Individual]] = None
    if size_objective == 'nsga2':
        print(f"{_tag} multi-objective mode: NSGA-II on (Dice, -{efficiency_metric}); "
              f"UCB/elite_k disabled.")
        population, best_ind, pareto_front = _evolve_nsga2(
            population=population,
            space=space,
            train_pool=train_pool,
            dataset_path=dataset_path,
            arch_dir=arch_dir,
            n_generations=n_generations,
            tournament_k=tournament_k,
            mutation_rate=mutation_rate,
            n_steps_init=n_steps_init,
            n_steps_offspring=n_steps_offspring,
            eps=eps,
            selection_strategy=selection_strategy,
            patience=patience,
            ind_cache=ind_cache,
            ind_cache_lock=ind_cache_lock,
            area_prior=area_prior,
            position_cache=global_position_cache,
            eval_logger=eval_logger,
            train_dataset=scoring_dataset,
            gpu_ids=_gpu_ids,
            n_parallel=n_parallel,
            no_image_inheritance=no_image_inheritance,
            post_processing=post_processing,
            output_folder=output_folder,
            best_ind_init=best_ind,
            efficiency_metric=efficiency_metric,
            coreset_image=coreset_image,
            color_space_str=color_space_str,
            use_filter_analysis=use_filter_analysis,
            tb_logger=tb_logger,
            max_score_candidates=max_score_candidates,
            score_images=score_images,
            best_stagnation_patience=best_stagnation_patience,
            min_mutation_rate=min_mutation_rate,
            max_mutation_rate=max_mutation_rate,
            mutation_schedule=mutation_schedule,
            checkpoint_path=checkpoint_path,
            ckpt_config=ckpt_config,
            start_gen=start_gen,
            resume_state=resume_state,
            **_adaptive_kw,
        )
    else:
        population, best_ind = _evolve(
            population=population,
            space=space,
            train_pool=train_pool,
            dataset_path=dataset_path,
            arch_dir=arch_dir,
            n_generations=n_generations,
            tournament_k=tournament_k,
            elite_k=elite_k,
            mutation_rate=mutation_rate,
            ucb_beta=ucb_beta,
            n_steps_init=n_steps_init,
            n_steps_offspring=n_steps_offspring,
            eps=eps,
            selection_strategy=selection_strategy,
            patience=patience,
            ind_cache=ind_cache,
            ind_cache_lock=ind_cache_lock,
            area_prior=area_prior,
            position_cache=global_position_cache,
            eval_logger=eval_logger,
            train_dataset=scoring_dataset,
            gpu_ids=_gpu_ids,
            n_parallel=n_parallel,
            no_image_inheritance=no_image_inheritance,
            post_processing=post_processing,
            output_folder=output_folder,
            best_ind_init=best_ind,
            coreset_image=coreset_image,
            color_space_str=color_space_str,
            use_filter_analysis=use_filter_analysis,
            tb_logger=tb_logger,
            max_score_candidates=max_score_candidates,
            score_images=score_images,
            best_stagnation_patience=best_stagnation_patience,
            reeval_elites=reeval_elites,
            tb_marker_log_every=tb_marker_log_every,
            min_mutation_rate=min_mutation_rate,
            max_mutation_rate=max_mutation_rate,
            mutation_schedule=mutation_schedule,
            checkpoint_path=checkpoint_path,
            ckpt_config=ckpt_config,
            start_gen=start_gen,
            resume_state=resume_state,
            **_adaptive_kw,
        )

    return GAResult(
        generation=n_generations,
        population=population,
        best_individual=best_ind,
        val_images=val_images,
        train_pool=train_pool,
        pareto_front=pareto_front,
    )
