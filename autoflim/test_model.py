import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io
from pyflim.datasets.flim_standard import FLIMStandardDataset
from torch.utils.data import DataLoader
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries
from .utils import keep_largest_component, filter_components_by_area, compute_area_prior
from .evaluation import _binarize
from pyflim.heads import HeadsRegistry

def get_args():
    parser = argparse.ArgumentParser(
        description="Test FLIM Encoder model and compute SOD metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-m', '--model_path',
        type=Path,
        required=True,
        help='Path to trained model (.pth file)'
    )
    
    parser.add_argument(
        '-d', '--dataset_path',
        type=Path,
        required=True,
        help='Path to test dataset folder'
    )
    
    parser.add_argument(
        '-s', '--split',
        type=int,
        required=True,
        help='Split number to test'
    )
    
    parser.add_argument(
        '-t', '--splits_data',
        type=Path,
        required=True,
        help='Path to folder with split data (test.txt)'
    )
    
    parser.add_argument(
        '-o', '--output_folder',
        type=Path,
        required=True,
        help='Path to output folder for results and metrics'
    )
    
    parser.add_argument(
        '-g', '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to use'
    )
    
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=16,
        help='Batch size for testing'
    )
    
    parser.add_argument(
        '-w', '--num_workers',
        type=int,
        default=8,
        help='Number of workers for data loading'
    )
    
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='Save prediction images'
    )
    
    parser.add_argument(
        '--save_overlays',
        action='store_true',
        help='Save overlay images (prediction + ground truth)'
    )

    parser.add_argument(
        '--train_percentage',
        type=str,
        default='100',
        help=(
            'Which train_<pct>pct.csv to use when recomputing the area prior '
            '(only used if area_prior.json is absent next to the model).'
        )
    )

    parser.add_argument(
        '--postprocess',
        choices=['auto', 'area_filter', 'largest_component', 'none'],
        default='auto',
        help=(
            'Post-processing strategy applied to raw predictions. '
            '"auto": use area_filter if prior is available, else largest_component. '
            '"area_filter": always use area prior filter. '
            '"largest_component": always keep only the largest connected component. '
            '"none": skip post-processing.'
        )
    )

    parser.add_argument(
        '--binarize_method',
        choices=['otsu', 'otsu_local', 'yen', 'sauvola'],
        default='otsu',
        help=(
            'Threshold used to binarize the decoder saliency map for headless '
            'models (head_type=none). Must match the value used during search. '
            '"otsu": global per-image Otsu. "yen": global per-image Yen. '
            '"otsu_local": per-pixel local Otsu over a disk neighborhood. '
            '"sauvola": per-pixel local Sauvola over a square window.'
        )
    )

    parser.add_argument(
        '--force_head',
        choices=HeadsRegistry.available(),
        default=None,
        help=(
            'Override the head baked into the loaded model. By default (None) '
            'the model uses whatever head the architecture search selected '
            '(the "arch head", which may be none/ift_segmentation/ca/...). '
            'When set, the named head replaces the model head before '
            'evaluation, so the same encoder/decoder is tested with a fixed '
            'segmentation head (e.g. "ift_segmentation").'
        )
    )

    parser.add_argument(
        '--force_head_erode_radius',
        type=float,
        default=8.0,
        help='erode_radius kwarg for --force_head (ift_segmentation / ca heads).'
    )

    parser.add_argument(
        '--force_head_seed_dilation',
        type=float,
        default=30.0,
        help='seed_dilation kwarg for --force_head (ift_segmentation head only).'
    )

    parser.add_argument(
        '--force_head_binarize_method',
        choices=['otsu', 'yen'],
        default='otsu',
        help=(
            'Threshold the ift_segmentation --force_head uses to turn the decoder '
            'saliency into its seed map. "yen" keeps larger, object-sized seeds '
            'than "otsu" (default), which survive the area prior better.'
        )
    )

    parser.add_argument(
        '--force_head_border_dist',
        type=int,
        default=-1,
        help=(
            'border_dist kwarg for --force_head (ift_segmentation head only). '
            'Seed components with a pixel within this many pixels of the image '
            'frame are dropped before delineation, mirroring iftRemoveCompNearFrame '
            'in iftSMansoniDelineation. -1 (default) disables border removal.'
        )
    )

    return parser.parse_args()


def compute_model_stats(model, sample_input, device, num_warmup=10, num_iters=50):
    """Compute parameter count, FLOPs (via FlopCounterMode), and inference FPS."""
    stats = {}

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats['total_params'] = total_params
    stats['trainable_params'] = trainable_params

    # FLOPs via eager FlopCounterMode (single image). Not fvcore's
    # FlopCountAnalysis: that jit-traces the model, and the FLIM forward has
    # data-dependent control flow that can abort tracing with an uncatchable
    # C++ std::terminate. FlopCounterMode counts ops in eager mode, so the
    # branches just run. See src/efficiency.py for the full rationale.
    try:
        from torch.utils.flop_counter import FlopCounterMode
        single = sample_input[:1]
        flop_counter = FlopCounterMode(display=False)
        model.eval()
        with torch.no_grad(), flop_counter:
            model(single)
        flops = int(flop_counter.get_total_flops())
        stats['flops'] = flops
        stats['gflops'] = flops / 1e9
    except Exception as e:
        stats['flops'] = None
        stats['gflops'] = None
        stats['flops_error'] = str(e)

    # FPS: warm up then time num_iters forward passes
    model.eval()
    batch_size = sample_input.shape[0]
    with torch.no_grad():
        for _ in range(num_warmup):
            model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    stats['fps'] = (num_iters * batch_size) / elapsed
    stats['fps_batch_size'] = batch_size

    return stats


def labnorm(image):
    """Normalize LAB images"""
    new_image = torch.empty_like(image, dtype=image.dtype)
    new_image[:, 0, :, :] = image[:, 0, :, :] / 99.998337
    new_image[:, 1, :, :] = (image[:, 1, :, :] + 86.182236) / (86.182236 + 98.258614)
    new_image[:, 2, :, :] = (image[:, 2, :, :] + 107.867744) / (107.867744 + 94.481682)
    return new_image



def compute_sod_metrics(prediction, ground_truth, epsilon=1e-7):
    """
    Compute comprehensive SOD (Salient Object Detection) metrics
    
    Args:
        prediction: Binary prediction mask (0 or 1)
        ground_truth: Binary ground truth mask (0 or 1)
        epsilon: Small value to avoid division by zero
    
    Returns:
        Dictionary containing all metrics
    """
    # Ensure binary masks
    pred = (prediction > 0).astype(np.float32)
    gt = (ground_truth > 0).astype(np.float32)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # Handle edge case: both masks are empty (all zeros)
    # This represents perfect agreement (both agree there's no object)
    if tp == 0 and fp == 0 and fn == 0:
        # Perfect agreement - both masks are empty
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
        f1_score = 1.0
    elif tp == 0 and (fp > 0 or fn > 0):
        # No true positives but there are false positives or false negatives
        dice = 0.0
        iou = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
    else:
        # Normal case with at least some true positives
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1_score = 2 * tp / (2 * tp + fp + fn + epsilon)
        dice = f1_score  # Dice coefficient is the same as F1
        iou = tp / (tp + fp + fn + epsilon)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    # Specificity
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 1.0 if fp == 0 else 0.0
    
    # Matthews Correlation Coefficient (MCC)
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = mcc_denominator = np.sqrt(np.float64(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if mcc_denominator > 0:
        mcc = mcc_numerator / mcc_denominator
    else:
        # If denominator is 0, check for perfect prediction
        if tp + fp + fn == 0:  # All true negatives
            mcc = 1.0
        else:
            mcc = 0.0
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(pred - gt))
    
    # Mean Squared Error (MSE)
    mse = np.mean((pred - gt) ** 2)
    
    # Structural Similarity Metrics
    # E-measure (Enhanced-alignment measure)
    e_measure = compute_e_measure(pred, gt)
    
    # S-measure (Structure measure)
    s_measure = compute_s_measure(pred, gt)
    
    # F-measure variants
    f_beta_scores = {}
    for beta in [0.3, 1.0, 2.0]:
        if tp == 0 and fp == 0 and fn == 0:
            # Perfect agreement - both empty
            f_beta = 1.0
        elif tp == 0 and (fp > 0 or fn > 0):
            # No true positives
            f_beta = 0.0
        else:
            f_beta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + epsilon)
        f_beta_scores[f'f{beta}_score'] = f_beta
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'specificity': specificity,
        'mcc': mcc,
        'mae': mae,
        'mse': mse,
        'e_measure': e_measure,
        's_measure': s_measure,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        **f_beta_scores
    }
    
    return metrics


def compute_e_measure(pred, gt, epsilon=1e-7):
    """Compute Enhanced-alignment measure (E-measure)"""
    # Ensure arrays are flattened
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Handle edge case: both masks are empty
    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        return 1.0  # Perfect agreement
    
    # Handle edge case: one mask is empty
    if np.sum(gt_flat) == 0 or np.sum(pred_flat) == 0:
        return 0.0
    
    # Mean values
    pred_mean = np.mean(pred_flat)
    gt_mean = np.mean(gt_flat)
    
    # Enhanced alignment matrix
    align_matrix = 2 * (pred_flat - pred_mean) * (gt_flat - gt_mean) / (
        (pred_flat - pred_mean)**2 + (gt_flat - gt_mean)**2 + epsilon
    )
    
    # Compute E-measure
    w = gt_mean
    fg_mask = gt_flat > 0.5
    bg_mask = gt_flat <= 0.5
    
    if np.sum(fg_mask) > 0 and np.sum(bg_mask) > 0:
        e_measure = w * np.mean(align_matrix[fg_mask]) + (1 - w) * np.mean(align_matrix[bg_mask])
    elif np.sum(fg_mask) > 0:
        e_measure = np.mean(align_matrix[fg_mask])
    elif np.sum(bg_mask) > 0:
        e_measure = np.mean(align_matrix[bg_mask])
    else:
        e_measure = 0
    
    return max(0, min(1, e_measure))  # Clip to [0, 1]


def compute_s_measure(pred, gt, alpha=0.5):
    """Compute Structure measure (S-measure)"""
    # Ensure arrays are flattened for proper indexing
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Handle edge case: both masks are empty
    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        return 1.0  # Perfect agreement
    
    # Handle edge case: one mask is empty
    if np.sum(gt_flat) == 0 or np.sum(pred_flat) == 0:
        return 0.0
    
    # Object-aware structural similarity
    gt_fg = (gt_flat > 0.5)
    gt_bg = (gt_flat <= 0.5)
    
    if np.sum(gt_fg) == 0:
        s_object = 0
    else:
        pred_fg = pred_flat[gt_fg]
        gt_fg_vals = gt_flat[gt_fg]
        s_object = np.mean(1 - np.abs(pred_fg - gt_fg_vals))
    
    if np.sum(gt_bg) == 0:
        s_region = 0
    else:
        pred_bg = pred_flat[gt_bg]
        gt_bg_vals = gt_flat[gt_bg]
        s_region = np.mean(1 - np.abs(pred_bg - gt_bg_vals))
    
    # Combine
    s_measure = alpha * s_object + (1 - alpha) * s_region
    
    return max(0, min(1, s_measure))  # Clip to [0, 1]

def save_overlay_image_with_contours(original, prediction, ground_truth, output_path):
    """
    Create overlay with contours of both prediction and ground truth
    
    Args:
        original: Original image (H, W, C) or (H, W) - may be in LAB color space
        prediction: Binary prediction (H, W)
        ground_truth: Binary ground truth (H, W)
        output_path: Path to save the overlay image
    """
    # Ensure images are in the right format
    if len(original.shape) == 2:
        original = gray2rgb(original)
    elif original.shape[2] > 3:
        original = original[:, :, :3]

    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8) if original.max() <= 1.0 else original.astype(np.uint8)

    overlay = original.copy()

    pred_mask = (prediction > 0)
    gt_mask = (ground_truth > 0)

    # Draw ground truth boundaries in magenta
    overlay[find_boundaries(gt_mask, mode='outer')] = [255, 0, 255]

    # Draw prediction boundaries in cyan
    overlay[find_boundaries(pred_mask, mode='outer')] = [0, 255, 255]

    io.imsave(str(output_path), overlay)

def save_overlay_image(original, prediction, ground_truth, output_path):
    """
    Create and save overlay visualization
    
    Args:
        original: Original image (H, W, C) or (H, W)
        prediction: Binary prediction (H, W)
        ground_truth: Binary ground truth (H, W)
        output_path: Path to save the overlay image
    """
    # Ensure images are in the right format
    if len(original.shape) == 2:
        original = gray2rgb(original)
    elif original.shape[2] > 3:
        original = original[:, :, :3]

    if original.max() <= 1.0:
        original = (original * 255).astype(np.uint8)
    else:
        original = original.astype(np.uint8)

    overlay = original.copy()

    # True Positives - Green
    tp_mask = (prediction > 0) & (ground_truth > 0)
    overlay[tp_mask] = (overlay[tp_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

    # False Positives - Red
    fp_mask = (prediction > 0) & (ground_truth == 0)
    overlay[fp_mask] = (overlay[fp_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    # False Negatives - Blue
    fn_mask = (prediction == 0) & (ground_truth > 0)
    overlay[fn_mask] = (overlay[fn_mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)

    io.imsave(str(output_path), overlay)


def test_model(args):
    """Main testing function"""
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    flim_predictions = output_folder / "flim_predictions"
    predictions_folder = output_folder / "predictions"
    overlays_folder = output_folder / "overlays"
    metrics_folder = output_folder / "metrics"
    
    if args.save_predictions:
        predictions_folder.mkdir(exist_ok=True)
        flim_predictions.mkdir(exist_ok=True)
    if args.save_overlays:
        overlays_folder.mkdir(exist_ok=True)
    metrics_folder.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # Load test split
    split_file = Path(args.splits_data) / f"split{args.split}" / f"test.csv"
    print(f"Loading test split from: {split_file}")
    
    with open(split_file, 'r') as f:
        test_images = [line.strip().replace('.png', '') for line in f.readlines()]
    
    print(f"Found {len(test_images)} test images")
    
    # Prepare dataset
    images_folder = args.dataset_path / "images"
    labels_folder = args.dataset_path / "truelabels"
    
    test_dataset = FLIMStandardDataset(
        image_folder=str(images_folder),
        images_list=test_images,
        label_folder=str(labels_folder),
        orig_ext=".png",
        label_ext="png",
        lab_norm=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Compute model stats (params, FLOPs, FPS) using the first batch
    print("\nComputing model statistics...")
    _sample_batch = next(iter(test_dataloader))
    _sample_input = _sample_batch['image'].to(device)
    model_stats = compute_model_stats(model, _sample_input, device)
    del _sample_batch, _sample_input

    # Optionally override the head the architecture search baked into the model.
    # Stats above are measured on the model as loaded (the "arch" pipeline); the
    # forced head is wired in afterwards so it only affects segmentation quality.
    if args.force_head is not None:
        head_kwargs = {
            'erode_radius': args.force_head_erode_radius,
            'seed_dilation': args.force_head_seed_dilation,
            'binarize_method': args.force_head_binarize_method,
            'border_dist': args.force_head_border_dist,
        }
        forced_head = HeadsRegistry.build(args.force_head, **head_kwargs)
        forced_head.to(device)
        model.head = forced_head
        print(f"Forced head '{args.force_head}' onto model "
              f"(erode_radius={args.force_head_erode_radius}, "
              f"seed_dilation={args.force_head_seed_dilation}, "
              f"binarize_method={args.force_head_binarize_method}, "
              f"border_dist={args.force_head_border_dist}).")


    print(f"  Total parameters:     {model_stats['total_params']:,}")
    print(f"  Trainable parameters: {model_stats['trainable_params']:,}")
    if model_stats['gflops'] is not None:
        print(f"  FLOPs (per image):    {model_stats['gflops']:.3f} GFLOPs")
    else:
        print(f"  FLOPs (per image):    N/A ({model_stats.get('flops_error', '')})")
    print(f"  Throughput (batch={model_stats['fps_batch_size']}): {model_stats['fps']:.1f} images/s")

    # Load area prior saved during search; fall back to recomputing from train split.
    area_prior = None, None
    saved_area_prior_path = args.model_path.parent / "area_prior.json"
    if saved_area_prior_path.exists():
        with open(saved_area_prior_path, 'r') as f:
            ap = json.load(f)
        area_prior = ap["lo"], ap["hi"]
        print(f"Area prior loaded from {saved_area_prior_path}: lo={area_prior[0]:.1f}  hi={area_prior[1]:.1f}")
    else:
        # Training lists follow the train_<pct>pct.csv convention produced by
        # `autoflim prepare`; --train_percentage selects which one.
        split_dir = Path(args.splits_data) / f"split{args.split}"
        train_file = split_dir / f"train_{args.train_percentage}pct.csv"
        if not train_file.exists():
            # Fall back to any training list in the split so the prior is still
            # recoverable when the percentage does not match exactly.
            candidates = sorted(split_dir.glob("train_*pct.csv"))
            if candidates:
                train_file = candidates[-1]
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_images = [line.strip().replace('.png', '') for line in f.readlines() if line.strip()]
            train_dataset = FLIMStandardDataset(
                image_folder=str(images_folder),
                images_list=train_images,
                label_folder=str(labels_folder),
                orig_ext=".png",
                label_ext="png",
                lab_norm=True,
            )
            area_prior = compute_area_prior(train_dataset)
            print(f"Area prior recomputed from training set: lo={area_prior[0]:.1f}  hi={area_prior[1]:.1f}")
        else:
            print(f"WARNING: no area_prior.json and train split not found at {train_file}, falling back to keep_largest_component")

    # Wire area_prior (computed from training images) into the model head so the
    # same thresholds used during the search are applied at test time.
    pp = args.postprocess
    head_has_area_prior = False
    if pp in ('area_filter', 'auto') and area_prior[0] is not None:
        head = getattr(model, 'head', None)
        if head is not None and hasattr(head, 'area_prior'):
            head.area_prior = area_prior
            head_has_area_prior = True
            print(f"Area prior wired into model head (consistent with search evaluation).")

    # Initialize metrics storage
    all_metrics = []
    image_names = []

    print("\nStarting testing...")
    print("="*80)

    with torch.no_grad():
        for sample in tqdm(test_dataloader, desc="Testing"):
            img = sample['image'].to(device)
            label = sample['label']
            label[label > 0] = 1
            filenames = [i.split('/')[-1] for i in sample['image_path']]

            res = model(img)[0].detach().cpu()

            # Models without a head output a raw saliency map. Binarize it per
            # image with the same method used during architecture search.
            if getattr(model, 'head', None) is None:
                res = _binarize(res, args.binarize_method)

            if not head_has_area_prior:
                # Head does not handle area filtering; apply external post-processing.
                if pp == 'area_filter' or (pp == 'auto' and area_prior[0] is not None):
                    res = filter_components_by_area(res, area_prior[0], area_prior[1]).numpy()
                elif pp == 'largest_component' or (pp == 'auto' and area_prior[0] is None):
                    res = keep_largest_component(res).numpy()
                else:  # none
                    res = res.numpy()
            else:
                res = res.numpy()

            for i in range(img.shape[0]):
                pred = res[i, 0, :, :]
                gt = label[i, 0, :, :].numpy()

                metrics = compute_sod_metrics(pred, gt)
                metrics['has_foreground'] = bool(gt.sum() > 0)
                metrics['image_name'] = filenames[i] if isinstance(filenames, list) else filenames
                all_metrics.append(metrics)
                image_names.append(metrics['image_name'])

                if args.save_predictions:
                    pred_path = predictions_folder / f"{metrics['image_name']}_pred.png"
                    Image.fromarray((pred * 255).astype(np.uint8)).save(pred_path)

                    flim_path = flim_predictions / f"{metrics['image_name']}_pred.png"
                    Image.fromarray((pred * 255).astype(np.uint8)).save(flim_path)

                if args.save_overlays:
                    original = np.array(Image.open(images_folder / filenames[i]))
                    overlay_path = overlays_folder / f"{metrics['image_name']}_overlay.png"
                    save_overlay_image_with_contours(original, pred, gt, overlay_path)
    
    print("\n" + "="*80)
    print("Testing completed!")
    print("="*80)
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    
    # --- Split into all-images and foreground-only subsets ---
    df_fg = df_metrics[df_metrics['has_foreground'] == True].copy()
    num_fg = len(df_fg)
    num_no_fg = len(df_metrics) - num_fg
    
    # Columns used for averaging (exclude non-numeric and the flag itself)
    numeric_columns = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
    if 'has_foreground' in numeric_columns:
        numeric_columns.remove('has_foreground')
    
    # Compute average/std for ALL images
    avg_metrics_all = df_metrics[numeric_columns].mean().to_dict()
    std_metrics_all = df_metrics[numeric_columns].std().to_dict()
    
    # Compute average/std for FOREGROUND-ONLY images
    if num_fg > 0:
        avg_metrics_fg = df_fg[numeric_columns].mean().to_dict()
        std_metrics_fg = df_fg[numeric_columns].std().to_dict()
    else:
        avg_metrics_fg = {col: float('nan') for col in numeric_columns}
        std_metrics_fg = {col: float('nan') for col in numeric_columns}
    
    # --- Helper to print a metrics block ---
    def print_metrics_block(avg, std):
        print(f"  Dice Score (F1):  {avg['dice']:.4f} ± {std['dice']:.4f}")
        print(f"  IoU:              {avg['iou']:.4f} ± {std['iou']:.4f}")
        print(f"  Precision:        {avg['precision']:.4f} ± {std['precision']:.4f}")
        print(f"  Recall:           {avg['recall']:.4f} ± {std['recall']:.4f}")
        print(f"  Accuracy:         {avg['accuracy']:.4f} ± {std['accuracy']:.4f}")
        print(f"  Specificity:      {avg['specificity']:.4f} ± {std['specificity']:.4f}")
        print(f"  MCC:              {avg['mcc']:.4f} ± {std['mcc']:.4f}")
        print(f"  MAE:              {avg['mae']:.4f} ± {std['mae']:.4f}")
        print(f"  E-measure:        {avg['e_measure']:.4f} ± {std['e_measure']:.4f}")
        print(f"  S-measure:        {avg['s_measure']:.4f} ± {std['s_measure']:.4f}")
    
    # --- Print summary: ALL images ---
    print("\n" + "="*80)
    print("RESULTS SUMMARY — ALL IMAGES")
    print("="*80)
    print(f"\nTotal test images: {len(df_metrics)}  "
          f"(with foreground: {num_fg}, without foreground: {num_no_fg})")
    print(f"\nAverage Metrics (all images):")
    print_metrics_block(avg_metrics_all, std_metrics_all)
    
    # --- Print summary: FOREGROUND-ONLY images ---
    print("\n" + "="*80)
    print("RESULTS SUMMARY — FOREGROUND-ONLY IMAGES")
    print("="*80)
    if num_fg > 0:
        print(f"\nNumber of images with foreground: {num_fg}")
        print(f"\nAverage Metrics (foreground-only):")
        print_metrics_block(avg_metrics_fg, std_metrics_fg)
    else:
        print("\n  No images with foreground pixels found in the test set.")
    
    # --- Save detailed metrics per image (includes has_foreground column) ---
    csv_path = metrics_folder / f"metrics_per_image_split{args.split}.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nDetailed metrics saved to: {csv_path}")
    
    # --- Save summary JSON (both all and foreground-only) ---
    summary_metrics = {
        'split': args.split,
        'num_images_total': len(df_metrics),
        'num_images_with_foreground': num_fg,
        'num_images_without_foreground': num_no_fg,
        'timestamp': datetime.now().isoformat(),
        'model_path': str(args.model_path),
        'model_stats': {
            'total_params': model_stats['total_params'],
            'trainable_params': model_stats['trainable_params'],
            'gflops_per_image': model_stats['gflops'],
            'fps': model_stats['fps'],
            'fps_batch_size': model_stats['fps_batch_size'],
        },
        'all_images': {
            'average_metrics': avg_metrics_all,
            'std_metrics': std_metrics_all
        },
        'foreground_only': {
            'average_metrics': avg_metrics_fg,
            'std_metrics': std_metrics_fg
        }
    }
    
    json_path = metrics_folder / f"summary_metrics_split{args.split}.json"
    with open(json_path, 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    print(f"Summary metrics saved to: {json_path}")
    
    # --- Save average metrics CSVs ---
    # All images
    avg_csv_path = metrics_folder / f"average_metrics_split{args.split}.csv"
    pd.DataFrame([avg_metrics_all]).to_csv(avg_csv_path, index=False)
    print(f"Average metrics (all) saved to: {avg_csv_path}")
    
    # Foreground-only
    avg_fg_csv_path = metrics_folder / f"average_metrics_fg_only_split{args.split}.csv"
    pd.DataFrame([avg_metrics_fg]).to_csv(avg_fg_csv_path, index=False)
    print(f"Average metrics (foreground-only) saved to: {avg_fg_csv_path}")
    
    print("\n" + "="*80)
    print("All results saved successfully!")
    print("="*80)


if __name__ == "__main__":
    args = get_args()
    test_model(args)