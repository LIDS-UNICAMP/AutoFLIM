import random
from statistics import median
import numpy as np
from PIL import Image


def select_coreset_initializer(available_images, img_path, color_space_str='lab'):
    """Return the image whose global color statistics are closest to the dataset mean.

    Provides a deterministic, data-driven first selection that replaces random
    episode initialization, eliminating the need for multiple episodes.
    """
    pil_mode = "LAB" if color_space_str == 'lab' else "L"
    img_means = {}
    for img_name in available_images:
        arr = np.array(
            Image.open(f"{img_path}/{img_name}.png").convert(pil_mode),
            dtype=np.float32,
        )
        img_means[img_name] = arr.reshape(-1, arr.shape[-1] if arr.ndim == 3 else 1).mean(axis=0)

    all_means = np.stack(list(img_means.values()))
    global_mean = all_means.mean(axis=0)
    return min(img_means, key=lambda n: float(np.linalg.norm(img_means[n] - global_mean)))


def select_candidate_image(unsolved_images, strategy='random',
                           previous_scores=None):
    """
    Select the next candidate image based on the chosen strategy.

    strategies: 'random', 'worst_first', 'median_first', 'diverse'
    """
    if len(unsolved_images) == 0:
        return None

    if strategy == 'random':
        return random.choice(unsolved_images)

    elif strategy == 'worst_first':
        if previous_scores is None or len(previous_scores) == 0:
            return random.choice(unsolved_images)
        # Unseen candidates (not yet in previous_scores) get -inf so they are
        # explored before images with known scores. This avoids the old bug where
        # images not in previous_scores were silently skipped and only surfaced as
        # a random fallback when none of the candidates had been seen.
        return min(unsolved_images,
                   key=lambda img: previous_scores.get(img, -float('inf')))

    elif strategy == 'median_first':
        if previous_scores is None or len(previous_scores) == 0:
            return random.choice(unsolved_images)
        scores = [(img, previous_scores.get(img, 0.0))
                  for img in unsolved_images if img in previous_scores]
        if not scores:
            return random.choice(unsolved_images)
        scores.sort(key=lambda x: x[1])
        median_idx = len(scores) // 2
        if median_idx >= len(scores):
            return random.choice(unsolved_images)
        return scores[median_idx][0]

    elif strategy == 'diverse':
        if previous_scores is None or len(previous_scores) == 0:
            return random.choice(unsolved_images)
        score_arr = np.array([previous_scores.get(img, 0.0) for img in unsolved_images])
        temperature = max(0.1, np.std(score_arr))
        neg_scores = -score_arr / temperature
        neg_scores -= neg_scores.max()
        weights = np.exp(neg_scores)
        weights /= weights.sum()
        # Use the seeded global RNG (np.random.seed is set in run_ga) rather than a
        # fresh default_rng(), which would draw from OS entropy and make 'diverse'
        # selection non-reproducible while every other selection path is seeded.
        idx = int(np.random.choice(len(unsolved_images), p=weights))
        return unsolved_images[idx]

    elif strategy == 'first_quartile':
        if previous_scores is None or len(previous_scores) == 0:
            return random.choice(unsolved_images)

        scores = [(img, previous_scores.get(img, 0.0))
                  for img in unsolved_images if img in previous_scores]

        if not scores:
            return random.choice(unsolved_images)

        scores.sort(key=lambda x: x[1])

        n = len(scores)
        q1 = n // 4

        # First quartile: candidates ranked in the bottom 25th percentile of
        # current Dice (the hardest images), but drawn at random from that band
        # rather than always the single lowest, which worst_first over-picks.
        first_quartile = scores[:q1]

        if not first_quartile:
            return random.choice(unsolved_images)

        return random.choice([img for img, _ in first_quartile])

    elif strategy == 'second_quartile':
        if previous_scores is None or len(previous_scores) == 0:
            return random.choice(unsolved_images)

        scores = [(img, previous_scores.get(img, 0.0))
                  for img in unsolved_images if img in previous_scores]

        if not scores:
            return random.choice(unsolved_images)

        scores.sort(key=lambda x: x[1])

        n = len(scores)
        q1 = n // 4
        median = n // 2

        # Second quartile: candidates ranked between the 25th and 50th
        # percentile of current Dice (just below the median) — harder-than-median
        # images, but not the lowest-Dice outliers that worst_first over-picks.
        second_quartile = scores[q1:median]

        if not second_quartile:
            return random.choice(unsolved_images)

        return random.choice([img for img, _ in second_quartile])

    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")


def select_error_targeted_image(
    available_images: list,
    images_folder: str,
    confusion_sig,
    color_space: str = 'lab',
    fallback_strategy: str = 'random',
    previous_scores: dict = None,
    model=None,
    labels_folder: str = None,
    device: str = None,
    max_score_candidates: float = -1,
) -> str:
    """
    Select the next candidate image guided by the current confusion signature.

    When model/labels_folder/device are provided, scores candidates by direct
    filter activation (confused + blind filters) — a feature-space signal.
    Otherwise falls back to Mahalanobis distance in color space.

    max_score_candidates: cap on how many remaining candidates are scored,
    instead of all of them (trades selection quality for speed).
    -1 scores all candidates (original behaviour). A value > 1 is an absolute
    count. A value in (0, 1] is a fraction of the remaining pool (min 1), so
    the subsample scales with the pool size.

    Falls back to fallback_strategy if no valid score can be computed.
    """
    if not available_images:
        return None

    if confusion_sig is None:
        return select_candidate_image(
            available_images, strategy=fallback_strategy,
            previous_scores=previous_scores,
        )

    candidates_to_score = available_images
    if max_score_candidates > 0:
        n_pool = len(available_images)
        if max_score_candidates <= 1:
            # Fraction of the remaining pool (at least 1 candidate).
            cap = max(1, int(np.ceil(max_score_candidates * n_pool)))
        else:
            # Absolute count.
            cap = int(max_score_candidates)
        if n_pool > cap:
            candidates_to_score = random.sample(available_images, cap)

    if model is not None and labels_folder is not None and device is not None:
        from .error_analysis import score_candidates_by_filter_response
        scores = score_candidates_by_filter_response(
            candidates_to_score, images_folder, labels_folder, model, confusion_sig, device,
        )
    else:
        from .error_analysis import score_image_by_confusion
        scores = {
            img: score_image_by_confusion(img, images_folder, confusion_sig, color_space)
            for img in candidates_to_score
        }

    # Pick the candidate whose confusion-signature match is strongest. Ties are
    # broken toward the LOWER current Dice (the harder image), consistent with the
    # error-targeting goal — the previous code capped scores at the 90th percentile
    # and broke ties toward easier images, which erased discrimination among the
    # best matches and contradicted the objective.
    best = max(scores, key=lambda x: (scores[x], -(previous_scores or {}).get(x, 1.0)))
    if scores[best] <= 0.0:
        return select_candidate_image(
            available_images, strategy=fallback_strategy,
            previous_scores=previous_scores,
        )
    return best
