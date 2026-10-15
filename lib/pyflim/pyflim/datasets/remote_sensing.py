import os
import re
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from . import DatasetRegistry
from .base import FLIMBaseDataset
from torchgeo.datasets import Potsdam2D, Vaihingen2D


# ---------------------------------------------------------------------------
# Stem helpers — must match generate_markers.py exactly
#
# full mode  : "tile_{tile_idx:05d}"
#              e.g. "tile_00005"
# patch mode : "tile_{tile_idx:05d}_r{row:04d}_c{col:04d}"
#              e.g. "tile_00005_r0256_c0512"
# ---------------------------------------------------------------------------

_PATCH_RE = re.compile(r"^tile_(\d+)_r(\d+)_c(\d+)$")


def make_full_stem(tile_idx: int) -> str:
    return f"tile_{tile_idx:05d}"


def make_patch_stem(tile_idx: int, row: int, col: int) -> str:
    return f"tile_{tile_idx:05d}_r{row:04d}_c{col:04d}"


def parse_patch_stem(stem: str) -> Optional[tuple[int, int, int]]:
    """
    Returns (tile_idx, row, col) if stem is a patch stem, else None.
    """
    m = _PATCH_RE.match(stem)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def parse_full_stem(stem: str) -> Optional[int]:
    """Returns tile_idx if stem is a full-tile stem, else None."""
    m = re.match(r"^tile_(\d+)$", stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Potsdam 2D Dataset
# ---------------------------------------------------------------------------

@DatasetRegistry.register("potsdam_2d")
class FLIMPotsdam2DDataset(FLIMBaseDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        marker_folder: str = None,
        marker_ext: str = "-seeds.txt",
        convert_to_lab: bool = False,
        lab_norm: bool = False,
        use_raw: bool = False,
        patch_size: int = None,
        patch_stride: int = None,
        stems: list = None,
        transforms=None,
    ):
        self.geo = Potsdam2D(root=root, split=split)
        self.marker_folder = marker_folder
        self.marker_ext = marker_ext
        self.convert_to_lab = convert_to_lab
        self.lab_norm = lab_norm
        self.use_raw = use_raw
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.transforms = transforms

        # If an explicit stem list is provided, use it directly.
        # Otherwise build the index from the full dataset.
        if stems is not None:
            self._index = self._index_from_stems(stems)
        elif patch_size is not None:
            self._index = self._build_patch_index()
        else:
            self._index = [(tile_idx,) for tile_idx in range(len(self.geo))]

    # -----------------------------------------------------------------------
    # Index helpers
    # -----------------------------------------------------------------------

    def _build_patch_index_for_tile(self, tile_idx: int) -> list:
        """Return (tile_idx, row, col) entries for all valid patches in one tile."""
        ps = self.patch_size
        st = self.patch_stride
        lbl_hw = self.geo[tile_idx]["mask"].numpy()
        H, W = lbl_hw.shape
        entries = []
        r = 0
        while r < H:
            c = 0
            while c < W:
                pr = min(r + ps, H)
                pc = min(c + ps, W)
                if lbl_hw[r:pr, c:pc].max() > 0:
                    entries.append((tile_idx, r, c))
                c += st
            r += st
        return entries

    def _build_patch_index(self) -> list:
        """
        Walk every tile and record (tile_idx, row, col) for every patch
        that contains at least one non-background pixel.
        """
        index = []
        for tile_idx in range(len(self.geo)):
            index.extend(self._build_patch_index_for_tile(tile_idx))
        return index

    def _index_from_stems(self, stems: list) -> list:
        """Convert a list of stems back to (tile_idx,) or (tile_idx, row, col).

        If patch_size is set and a stem is a full-tile stem, the tile is expanded
        into its patch entries so the DataLoader always receives patch-sized tensors.
        """
        index = []
        for stem in stems:
            parsed = parse_patch_stem(stem)
            if parsed is not None:
                index.append(parsed)          # (tile_idx, row, col)
            else:
                tile_idx = parse_full_stem(stem)
                if tile_idx is None:
                    raise ValueError(f"Cannot parse stem: '{stem}'")
                if self.patch_size is not None:
                    # Full-tile stem + patch mode → expand to patch entries
                    index.extend(self._build_patch_index_for_tile(tile_idx))
                else:
                    index.append((tile_idx,))     # (tile_idx,)
        return index

    # -----------------------------------------------------------------------
    # Dataset protocol
    # -----------------------------------------------------------------------

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        entry = self._index[idx]
        if len(entry) == 1:
            return self._getitem_full(entry[0])
        else:
            return self._getitem_patch(*entry)

    # -----------------------------------------------------------------------
    # Full-tile item (original behaviour, unchanged)
    # -----------------------------------------------------------------------

    def _getitem_full(self, tile_idx: int):
        geo_sample = self.geo[tile_idx]
        raw = geo_sample["image"]

        if self.use_raw:
            image = raw.float()
        else:
            if self.convert_to_lab:
                image = raw[:3].int()
                image = (
                    super().image_to_lab_norm(image)
                    if self.lab_norm
                    else super().image_to_lab(image)
                )
            else:
                image = raw.float()

        stem = make_full_stem(tile_idx)
        original_size = torch.tensor(list(image.shape[1:]))  # (H, W)

        sample = {
            "image":         image,
            "image_path":    stem,
            "original_size": original_size,
        }

        if "mask" in geo_sample:
            sample["label"] = geo_sample["mask"].unsqueeze(0)

        if self.marker_folder is not None:
            marker_file = os.path.join(self.marker_folder, stem + self.marker_ext)
            if os.path.isfile(marker_file):
                marker_image = super().get_marker_image(marker_file)
                sample["markers"]       = marker_image
                sample["markers_label"] = super().label_markers_by_component(marker_image)

        return sample

    # -----------------------------------------------------------------------
    # Patch item — cuts on-the-fly, stem encodes tile+position
    # -----------------------------------------------------------------------

    def _getitem_patch(self, tile_idx: int, row: int, col: int):
        ps = self.patch_size
        geo_sample = self.geo[tile_idx]
        raw = geo_sample["image"]          # (C, H, W)
        C, H, W = raw.shape
        pr = min(row + ps, H)
        pc = min(col + ps, W)

        # Cut image patch, zero-pad at borders
        raw_patch = torch.zeros(C, ps, ps, dtype=raw.dtype)
        raw_patch[:, :pr - row, :pc - col] = raw[:, row:pr, col:pc]

        if self.use_raw:
            image = raw_patch.float()
        else:
            if self.convert_to_lab:
                image = raw_patch[:3].int()
                image = (
                    super().image_to_lab_norm(image)
                    if self.lab_norm
                    else super().image_to_lab(image)
                )
            else:
                image = raw_patch.float()

        stem = make_patch_stem(tile_idx, row, col)
        original_size = torch.tensor([ps, ps])

        sample = {
            "image":         image,
            "image_path":    stem,
            "original_size": original_size,
        }

        if "mask" in geo_sample:
            lbl_patch = torch.zeros(ps, ps, dtype=geo_sample["mask"].dtype)
            lbl_patch[:pr - row, :pc - col] = geo_sample["mask"][row:pr, col:pc]
            sample["label"] = lbl_patch.unsqueeze(0)

        if self.marker_folder is not None:
            marker_file = os.path.join(self.marker_folder, stem + self.marker_ext)
            if os.path.isfile(marker_file):
                marker_image = super().get_marker_image(marker_file)
                sample["markers"]       = marker_image
                sample["markers_label"] = super().label_markers_by_component(marker_image)

        return sample