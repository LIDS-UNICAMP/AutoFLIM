import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .genome import SearchSpace, is_valid_genome


@dataclass
class Individual:
    """
    A GA individual: architecture genome + greedily-selected training set.

    image_set evolves alongside the genome via warm-start greedy refinement.
    mean_score is the fitness signal; n_evals counts evaluations and drives
    UCB1 exploration (UCB = mean + beta * sqrt(2*ln(total) / n_evals)).
    std_score retains per-image Dice spread for logging only.
    per_image_scores is kept for worst_first candidate selection in the next
    greedy refinement step.
    image_gains maps each training image to its marginal Dice gain when it was
    accepted by dynamic_greedy_refine (inf for inherited images); used to rank training
    images by importance when max_selected_images truncation is active.
    """
    genome: dict
    image_set: List[str] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 1.0
    n_evals: int = 0
    per_image_scores: Dict[str, float] = field(default_factory=dict)
    image_gains: Dict[str, float] = field(default_factory=dict)
    # Multi-objective (NSGA-II) bookkeeping. flops/n_params measure the final
    # model's compute/size (flops None when it couldn't be measured). rank and
    # crowding are assigned per-generation by the non-dominated sort and are
    # only meaningful in size_objective='nsga2' mode.
    flops: Optional[int] = None
    n_params: Optional[int] = None
    rank: int = 0
    crowding: float = 0.0

    @property
    def fitness(self) -> float:
        return self.mean_score

    def size_cost(self, metric: str = 'flops') -> float:
        """Size objective to MINIMISE: FLOPs (preferred) or param count.

        Falls back to n_params when flops is unavailable. A model that never
        trained (both None) costs +inf so NSGA-II ranks it last.
        """
        val = self.flops if metric == 'flops' else self.n_params
        if val is None:
            val = self.n_params if metric == 'flops' else self.flops
        return float(val) if val is not None else float('inf')

    def cache_key(self) -> str:
        import json
        return json.dumps(
            {**self.genome, '__imgs__': sorted(self.image_set)},
            sort_keys=True,
        )


def tournament_select(
    population: List[dict],
    scores: List[float],
    k: int = 3,
) -> dict:
    """k-way tournament selection; higher score wins."""
    contestants = random.sample(range(len(population)), min(k, len(population)))
    winner = max(contestants, key=lambda i: scores[i])
    return dict(population[winner])


def ucb_select(
    population: List[dict],
    means: List[float],
    stds: List[float],
    k: int = 3,
    beta: float = 0.5,
) -> dict:
    """
    Tournament selection using UCB = mean + beta * std.
    beta=0 → pure exploitation; beta=1 → strong exploration.
    """
    ucb = [m + beta * s for m, s in zip(means, stds)]
    contestants = random.sample(range(len(population)), min(k, len(population)))
    winner = max(contestants, key=lambda i: ucb[i])
    return dict(population[winner])


def genome_crossover(parent_a: dict, parent_b: dict) -> dict:
    """Uniform crossover: each gene is independently inherited from one parent."""
    return {
        gene: (parent_a[gene] if random.random() < 0.5 else parent_b[gene])
        for gene in parent_a
    }


def random_mutate(
    genome: dict,
    space: SearchSpace,
    mutation_rate: float = 0.2,
) -> dict:
    """Per-gene random mutation. Always returns a genome with valid search-space values."""
    new_genome = dict(genome)
    for gene, options in space.as_dict().items():
        if random.random() < mutation_rate:
            new_genome[gene] = random.choice(options)
    return new_genome



def adapt_mutation_rate(
    mutation_rate: float,
    recent_successes: int,
    recent_trials: int,
    target_rate: float = 0.2,
    increase_factor: float = 1.15,
    decrease_factor: float = 0.90,
    min_rate: float = 0.05,
    max_rate: float = 0.3,
) -> float:
    """
    1/5-success-rule adaptation of the per-gene mutation rate.

    Compares the fraction of offspring that improved over the previous
    generation's mean to a target success rate (default 0.2 = 1/5):
      - success_ratio > target: more offspring improved than expected → decrease
        the rate to exploit the productive region of the search space.
      - success_ratio < target: too few improved → increase the rate to escape
        the current neighbourhood.
      - success_ratio ≈ target: no change.

    Bounds are clamped to [min_rate, max_rate] to prevent degenerate behaviour
    (rate → 0 stalls diversity; rate → 1 becomes random restart).
    """
    if recent_trials == 0:
        return mutation_rate
    success_ratio = recent_successes / recent_trials
    if success_ratio > target_rate:
        return max(min_rate, mutation_rate * decrease_factor)
    elif success_ratio < target_rate:
        return min(max_rate, mutation_rate * increase_factor)
    return mutation_rate


def get_elite(
    population: List[dict],
    scores: List[float],
    k: int = 2,
) -> Tuple[List[dict], List[float]]:
    """Return the top-k genomes and their scores, sorted best-first."""
    ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    elite_pop, elite_scores = zip(*ranked[:k])
    return list(elite_pop), list(elite_scores)


# ---------------------------------------------------------------------------
# Individual-level helpers
# ---------------------------------------------------------------------------

def get_elite_individuals(
    population: List['Individual'],
    k: int = 2,
) -> List['Individual']:
    """Return the top-k Individual objects by mean_score, best-first."""
    return sorted(population, key=lambda ind: ind.mean_score, reverse=True)[:k]


def ucb_select_individual(
    population: List['Individual'],
    k: int = 3,
    beta: float = 0.5,
) -> 'Individual':
    """UCB1 tournament selection over Individual objects.

    UCB = mean + beta * sigma * sqrt(2*ln(total) / n_i), where sigma is the
    std of mean_score across the population. Scaling the exploration bonus by
    sigma keeps beta unitless and on the same scale as fitness: the raw
    sqrt(2*ln(total)/n) term is ~1.9 for a fresh offspring, which on its own
    dwarfs mean_score in [0,1] and turns selection into pure novelty search.
    With the sigma factor, exploration is proportional to the actual fitness
    spread and fades to 0 as the population converges (sigma -> 0), so higher
    fitness is genuinely preferred. Unevaluated individuals (n_evals=0) still
    receive the maximum bonus.
    """
    total = max(1, sum(ind.n_evals for ind in population))
    log_total = math.log(total)
    sigma = float(np.std([ind.mean_score for ind in population])) if population else 0.0

    def _ucb(ind: 'Individual') -> float:
        n = ind.n_evals if ind.n_evals > 0 else 1
        return ind.mean_score + beta * sigma * math.sqrt(2 * log_total / n)

    contestants = random.sample(range(len(population)), min(k, len(population)))
    winner = max(contestants, key=lambda i: _ucb(population[i]))
    return population[winner]


# ---------------------------------------------------------------------------
# NSGA-II multi-objective selection: maximise Dice, minimise size (FLOPs/params)
# ---------------------------------------------------------------------------

def _dominates(a: 'Individual', b: 'Individual', metric: str) -> bool:
    """True iff a Pareto-dominates b on (Dice maximise, size minimise).

    a dominates b when it is no worse on both objectives and strictly better on
    at least one.
    """
    a_d, b_d = a.mean_score, b.mean_score
    a_s, b_s = a.size_cost(metric), b.size_cost(metric)
    no_worse = (a_d >= b_d) and (a_s <= b_s)
    strictly_better = (a_d > b_d) or (a_s < b_s)
    return no_worse and strictly_better


def fast_non_dominated_sort(
    population: List['Individual'],
    metric: str = 'flops',
) -> List[List[int]]:
    """Deb's fast non-dominated sort.

    Returns a list of fronts, each a list of indices into ``population``,
    ordered best (front 0 = Pareto-optimal set) to worst. Also writes each
    individual's ``rank`` (front index) as a side effect.
    """
    n = len(population)
    dominated: List[List[int]] = [[] for _ in range(n)]   # solutions p dominates
    dom_count = [0] * n                                   # how many dominate p
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(population[p], population[q], metric):
                dominated[p].append(q)
            elif _dominates(population[q], population[p], metric):
                dom_count[p] += 1
        if dom_count[p] == 0:
            population[p].rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    population[q].rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)

    fronts.pop()  # drop the trailing empty front
    return fronts


def assign_crowding_distance(
    population: List['Individual'],
    front: List[int],
    metric: str = 'flops',
) -> None:
    """Assign NSGA-II crowding distance to every individual in one front.

    Boundary solutions (best/worst on an objective) get +inf so they are always
    preserved; interior solutions get the sum of normalised neighbour gaps across
    both objectives. Writes ``crowding`` on each individual.
    """
    if not front:
        return
    for idx in front:
        population[idx].crowding = 0.0
    if len(front) <= 2:
        for idx in front:
            population[idx].crowding = float('inf')
        return

    objectives = [
        lambda ind: ind.mean_score,            # maximise
        lambda ind: ind.size_cost(metric),     # minimise
    ]
    for obj in objectives:
        ordered = sorted(front, key=lambda i: obj(population[i]))
        lo = obj(population[ordered[0]])
        hi = obj(population[ordered[-1]])
        rng = hi - lo
        population[ordered[0]].crowding = float('inf')
        population[ordered[-1]].crowding = float('inf')
        if rng <= 0:
            continue
        for k in range(1, len(ordered) - 1):
            prev_v = obj(population[ordered[k - 1]])
            next_v = obj(population[ordered[k + 1]])
            population[ordered[k]].crowding += (next_v - prev_v) / rng


def crowded_compare(a: 'Individual', b: 'Individual') -> 'Individual':
    """NSGA-II crowded-comparison operator: lower rank wins; ties → larger
    crowding distance wins (favours less crowded regions of the front)."""
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding >= b.crowding else b


def nsga2_tournament_select(
    population: List['Individual'],
    k: int = 2,
) -> 'Individual':
    """Binary (k-way) tournament using the crowded-comparison operator.

    Assumes rank/crowding have been assigned for the current population.
    """
    contestants = random.sample(range(len(population)),
                                min(k, len(population)))
    winner = population[contestants[0]]
    for i in contestants[1:]:
        winner = crowded_compare(winner, population[i])
    return winner


def nsga2_select_next_generation(
    combined: List['Individual'],
    pop_size: int,
    metric: str = 'flops',
) -> List['Individual']:
    """NSGA-II (mu+lambda) survivor selection.

    Sorts the combined parent+offspring set into fronts and fills the next
    generation front by front; the last partially-included front is truncated by
    descending crowding distance to preserve spread. Also (re)assigns rank and
    crowding on the survivors so the next round's tournaments are valid.
    """
    fronts = fast_non_dominated_sort(combined, metric)
    survivors: List['Individual'] = []
    for front in fronts:
        assign_crowding_distance(combined, front, metric)
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(combined[i] for i in front)
        else:
            remaining = pop_size - len(survivors)
            ordered = sorted(front, key=lambda i: combined[i].crowding,
                             reverse=True)
            survivors.extend(combined[i] for i in ordered[:remaining])
            break
    return survivors


def pareto_front_individuals(
    population: List['Individual'],
    metric: str = 'flops',
) -> List['Individual']:
    """Return the non-dominated (front 0) individuals, best Dice first."""
    fronts = fast_non_dominated_sort(population, metric)
    if not fronts:
        return []
    front0 = [population[i] for i in fronts[0]]
    return sorted(front0, key=lambda ind: ind.mean_score, reverse=True)


def crossover_individuals(parent_a: 'Individual', parent_b: 'Individual') -> dict:
    """
    Depth-aware uniform genome crossover between two individuals.

    Non-layer genes are inherited uniformly (each gene from either parent with
    p=0.5). The per-layer genes (kernel_size_i, pool_type_i) are treated as a
    unit and only filled for layers WITHIN the child's inherited depth:

      - The child's n_layers is inherited uniformly like any scalar gene.
      - For each active layer i (1..n_layers), the (kernel_size_i, pool_type_i)
        pair is taken together from a parent that actually has a real (non-None)
        gene there. When both parents reach layer i, the pair is picked from one
        at random; when only one is deep enough, its pair is used. Because
        n_layers comes from one of the parents, that parent always has real genes
        up to n_layers, so an active layer can never end up None.
      - Layers beyond n_layers are set to None (already canonical).

    This guarantees a valid, already-canonical child for ANY pair of parent
    depths. Plain uniform per-gene crossover could not: two equally-shallow
    parents share None in the deepest slots, so the child inherited None there
    and was always rejected by is_valid_genome — making recombination between
    same-depth (sub-maximal) architectures impossible (they could only persist
    via mutation/elitism/injection, never recombine).

    Returns a child genome (dict); the image_set is assigned separately by
    run_ga (inherited from the better-scoring parent).
    """
    ga, gb = parent_a.genome, parent_b.genome
    n_slots = sum(1 for k in ga if k.startswith('kernel_size_'))

    child: dict = {}
    for gene in ga:
        if gene.startswith(('kernel_size_', 'pool_type_')):
            continue
        child[gene] = ga[gene] if random.random() < 0.5 else gb[gene]

    n = child.get('n_layers', 1)
    for i in range(1, n_slots + 1):
        if i > n:
            child[f'kernel_size_{i}'] = None
            child[f'pool_type_{i}'] = None
            continue
        a_ks, b_ks = ga.get(f'kernel_size_{i}'), gb.get(f'kernel_size_{i}')
        a_ok, b_ok = a_ks is not None, b_ks is not None
        take_a = (random.random() < 0.5) if (a_ok and b_ok) else a_ok
        if take_a:
            child[f'kernel_size_{i}'] = a_ks
            child[f'pool_type_{i}'] = ga.get(f'pool_type_{i}')
        else:
            child[f'kernel_size_{i}'] = b_ks
            child[f'pool_type_{i}'] = gb.get(f'pool_type_{i}')
    return child
