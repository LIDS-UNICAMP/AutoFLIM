import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description="Combine test results from multiple splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input_folder',
        type=Path,
        required=True,
        help='Base folder containing split results'
    )
    
    parser.add_argument(
        '-s', '--splits',
        nargs='+',
        type=int,
        default=None,
        help='Split numbers to combine (e.g., 1 2 3). '
             'If omitted, every split<N> folder with a metrics file is auto-detected.'
    )

    return parser.parse_args()


def discover_splits(input_folder):
    """Return the sorted list of split numbers that actually have a metrics CSV
    on disk under input_folder/split<N>/metrics/metrics_per_image_split<N>.csv."""
    input_folder = Path(input_folder)
    found = []
    for csv_path in input_folder.glob("split*/metrics/metrics_per_image_split*.csv"):
        try:
            split = int(csv_path.parent.parent.name.replace("split", ""))
        except ValueError:
            continue
        found.append(split)
    return sorted(set(found))


def combine_results(input_folder, splits):
    """
    Combine metrics from multiple splits into aggregate statistics.
    Supports the has_foreground column produced by the modified test_model.py;
    gracefully skips foreground-only stats if the column is absent.

    Args:
        input_folder: Path to base folder containing split subfolders
        splits: List of split numbers to combine

    Returns:
        combined_df: DataFrame with all metrics from all splits
        combined_df_fg: DataFrame filtered to foreground-only rows (None if column missing)
        overall_stats: Dictionary with aggregate statistics for both subsets
    """

    input_folder = Path(input_folder)

    print("="*80)
    print("COMBINING RESULTS FROM MULTIPLE SPLITS")
    print("="*80)

    # Combine per-image metrics from all splits
    all_metrics = []
    splits_found = []
    splits_missing = []

    for split in splits:
        csv_path = input_folder / f"split{split}" / "metrics" / f"metrics_per_image_split{split}.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['split'] = split
            all_metrics.append(df)
            splits_found.append(split)
            print(f"  Loaded split {split}: {len(df)} images")
        else:
            splits_missing.append(split)
            print(f"  Split {split} not found at: {csv_path}")

    if not all_metrics:
        # Raise rather than exit: this is a library function, and callers
        # (the CLI, the test driver) decide how to report the failure.
        raise FileNotFoundError(
            f"No per-split metrics CSVs found under {input_folder} for splits "
            f"{list(splits)}. Run the test step for at least one split first."
        )

    if splits_missing:
        print(f"\n  Warning: {len(splits_missing)} split(s) missing: {splits_missing}")

    # Combine all splits
    combined_df = pd.concat(all_metrics, ignore_index=True)

    print(f"\n  Combined {len(combined_df)} total images from {len(splits_found)} splits")

    # Determine metric columns (exclude bookkeeping columns)
    numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ('split', 'has_foreground')]

    # --- Detect has_foreground column (produced by modified test_model.py) ---
    has_fg_column = 'has_foreground' in combined_df.columns

    if has_fg_column:
        combined_df_fg = combined_df[combined_df['has_foreground'] == True].copy()
        combined_df_bg = combined_df[combined_df['has_foreground'] == False].copy()
        num_fg    = len(combined_df_fg)
        num_no_fg = len(combined_df_bg)
        print(f"  Foreground split: {num_fg} with foreground, {num_no_fg} without foreground")
    else:
        combined_df_fg = None
        combined_df_bg = None
        num_fg         = None
        num_no_fg      = None
        print("  has_foreground column not found — skipping foreground/background-only stats")

    # --- Helper: compute full stat block for a DataFrame ---
    def round_dict(d):
        return {k: round(v, 2) for k, v in d.items()}

    def compute_stats(df):
        return {
            'num_images': len(df),
            'mean':   round_dict(df[numeric_cols].mean().to_dict()),
            'std':    round_dict(df[numeric_cols].std().to_dict()),
            'median': round_dict(df[numeric_cols].median().to_dict()),
            'min':    round_dict(df[numeric_cols].min().to_dict()),
            'max':    round_dict(df[numeric_cols].max().to_dict())
        }

    # --- Top-level structure ---
    overall_stats = {
        'num_splits': len(splits_found),
        'splits_included': splits_found,
        'total_images': len(combined_df),
        'has_foreground_column': has_fg_column,
        'all_images': compute_stats(combined_df)
    }

    # --- Foreground-only / background-only stats (top level) ---
    # Background-only = empty-GT images: in a real screening scenario most fields
    # are negative, so the model's behaviour here (false positives, specificity)
    # matters as much as foreground dice. Image-level dice on this subset equals
    # the fraction of empty images left correctly empty (1.0 = no false positive).
    if has_fg_column:
        overall_stats['num_images_with_foreground']    = num_fg
        overall_stats['num_images_without_foreground'] = num_no_fg
        overall_stats['foreground_only'] = compute_stats(combined_df_fg) if num_fg > 0 else None
        overall_stats['background_only'] = compute_stats(combined_df_bg) if num_no_fg > 0 else None

    # --- Per-split stats (all images + foreground-only where available) ---
    per_split_stats = {}
    for split in splits_found:
        split_data = combined_df[combined_df['split'] == split]
        split_entry = {
            'num_images': len(split_data),
            'mean': round_dict(split_data[numeric_cols].mean().to_dict()),
            'std':  round_dict(split_data[numeric_cols].std().to_dict())
        }

        if has_fg_column:
            split_fg = split_data[split_data['has_foreground'] == True]
            split_bg = split_data[split_data['has_foreground'] == False]
            split_entry['num_images_with_foreground']    = len(split_fg)
            split_entry['num_images_without_foreground'] = len(split_bg)
            if len(split_fg) > 0:
                split_entry['foreground_only'] = {
                    'mean': round_dict(split_fg[numeric_cols].mean().to_dict()),
                    'std':  round_dict(split_fg[numeric_cols].std().to_dict())
                }
            else:
                split_entry['foreground_only'] = None
            if len(split_bg) > 0:
                split_entry['background_only'] = {
                    'mean': round_dict(split_bg[numeric_cols].mean().to_dict()),
                    'std':  round_dict(split_bg[numeric_cols].std().to_dict())
                }
            else:
                split_entry['background_only'] = None

        per_split_stats[f'split_{split}'] = split_entry

    overall_stats['per_split'] = per_split_stats

    return combined_df, combined_df_fg, combined_df_bg, overall_stats


def save_results(combined_df, combined_df_fg, combined_df_bg, overall_stats, output_folder):
    """Save combined results and statistics"""

    output_folder = Path(output_folder)

    # Save combined metrics CSV (all images, includes has_foreground if present)
    combined_csv = output_folder / "all_splits_metrics.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n  Saved combined metrics (all):            {combined_csv}")

    # Save foreground-only combined CSV if available
    if combined_df_fg is not None and len(combined_df_fg) > 0:
        fg_csv = output_folder / "all_splits_metrics_fg_only.csv"
        combined_df_fg.to_csv(fg_csv, index=False)
        print(f"  Saved combined metrics (foreground only): {fg_csv}")

    # Save background-only (empty-GT) combined CSV if available
    if combined_df_bg is not None and len(combined_df_bg) > 0:
        bg_csv = output_folder / "all_splits_metrics_bg_only.csv"
        combined_df_bg.to_csv(bg_csv, index=False)
        print(f"  Saved combined metrics (background only): {bg_csv}")

    # Save overall statistics JSON
    stats_json = output_folder / "overall_statistics.json"
    with open(stats_json, 'w') as f:
        json.dump(overall_stats, f, indent=4)
    print(f"  Saved overall statistics:                {stats_json}")

    return combined_csv, stats_json


def print_summary(overall_stats):
    """Print summary statistics to console for both all-images and foreground-only"""

    key_metrics = ['dice', 'iou', 'precision', 'recall', 'accuracy', 'specificity',
                   'mcc', 'mae', 'e_measure', 's_measure']

    def print_metrics_block(stats_block):
        for metric in key_metrics:
            if metric in stats_block['mean']:
                mean_val = stats_block['mean'][metric]
                std_val  = stats_block['std'][metric]
                print(f"    {metric.upper():15s}: {mean_val:.2f} +/- {std_val:.2f}")

    has_fg = overall_stats.get('has_foreground_column', False)

    # ---------- ALL IMAGES ----------
    print("\n" + "="*80)
    print("OVERALL RESULTS — ALL IMAGES (All Splits Combined)")
    print("="*80)
    print(f"\n  Total images tested : {overall_stats['total_images']}")
    print(f"  Number of splits    : {overall_stats['num_splits']}")
    print(f"  Splits included     : {overall_stats['splits_included']}")
    if has_fg:
        print(f"  With foreground     : {overall_stats['num_images_with_foreground']}")
        print(f"  Without foreground  : {overall_stats['num_images_without_foreground']}")

    print(f"\n  Average Metrics (all images):")
    print("  " + "-"*60)
    print_metrics_block(overall_stats['all_images'])

    # ---------- FOREGROUND-ONLY ----------
    if has_fg:
        print("\n" + "="*80)
        print("OVERALL RESULTS — FOREGROUND-ONLY IMAGES (All Splits Combined)")
        print("="*80)
        if overall_stats['foreground_only'] is not None:
            print(f"\n  Number of images with foreground: {overall_stats['num_images_with_foreground']}")
            print(f"\n  Average Metrics (foreground only):")
            print("  " + "-"*60)
            print_metrics_block(overall_stats['foreground_only'])
        else:
            print("\n  No images with foreground pixels found across all splits.")

    # ---------- BACKGROUND-ONLY (empty-GT) ----------
    # On empty-GT images precision/recall/iou are degenerate (no true positives
    # possible), so only the metrics that describe correct rejection are printed:
    # dice here = fraction of empty fields left empty (1.0 = no false positive).
    if has_fg:
        print("\n" + "="*80)
        print("OVERALL RESULTS — BACKGROUND-ONLY / EMPTY-GT IMAGES (All Splits Combined)")
        print("="*80)
        if overall_stats.get('background_only') is not None:
            bg = overall_stats['background_only']
            print(f"\n  Number of images without foreground: {overall_stats['num_images_without_foreground']}")
            print(f"\n  Average Metrics (background only):")
            print("  " + "-"*60)
            for metric in ['dice', 'specificity', 'accuracy', 'mcc', 'mae']:
                if metric in bg['mean']:
                    print(f"    {metric.upper():15s}: {bg['mean'][metric]:.2f} +/- {bg['std'][metric]:.2f}")
        else:
            print("\n  No empty-GT images found across all splits.")

    # ---------- PER-SPLIT COMPARISON ----------
    print("\n" + "="*80)
    print("Per-Split Comparison")
    print("="*80)

    for split_key in sorted(overall_stats['per_split'].keys()):
        split_stats = overall_stats['per_split'][split_key]
        split_num   = split_key.replace('split_', '')

        # All images for this split
        print(f"\n  Split {split_num} — all images (n={split_stats['num_images']}):")
        for metric in ['dice', 'iou', 'precision', 'recall']:
            if metric in split_stats['mean']:
                mean_val = split_stats['mean'][metric]
                std_val  = split_stats['std'][metric]
                print(f"    {metric.upper():12s}: {mean_val:.2f} +/- {std_val:.2f}")

        # Foreground-only for this split
        if has_fg and split_stats.get('foreground_only') is not None:
            fg = split_stats['foreground_only']
            print(f"\n  Split {split_num} — foreground only (n={split_stats['num_images_with_foreground']}):")
            for metric in ['dice', 'iou', 'precision', 'recall']:
                if metric in fg['mean']:
                    mean_val = fg['mean'][metric]
                    std_val  = fg['std'][metric]
                    print(f"    {metric.upper():12s}: {mean_val:.2f} +/- {std_val:.2f}")

    print("\n" + "="*80)


def main():
    args = get_args()

    # Resolve which splits to combine: an explicit list if given, otherwise
    # every split present on disk so previously-tested splits are never dropped.
    splits = args.splits
    if splits is None:
        splits = discover_splits(args.input_folder)
        if not splits:
            print(f"ERROR: no split<N>/metrics CSVs found under {args.input_folder}")
            sys.exit(1)
        print(f"Auto-detected splits on disk: {splits}")

    # Combine results
    combined_df, combined_df_fg, combined_df_bg, overall_stats = combine_results(
        args.input_folder, splits)

    # Save results
    save_results(combined_df, combined_df_fg, combined_df_bg, overall_stats, args.input_folder)

    # Print summary
    print_summary(overall_stats)

    print(f"\n  Results successfully combined and saved to: {args.input_folder}")


if __name__ == "__main__":
    main()