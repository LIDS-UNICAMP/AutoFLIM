import os
import torch
import numpy as np
from skimage import io, measure
import torch.nn.functional as F
import xml.etree.ElementTree as ET
from skimage.filters import threshold_otsu
from torchvision.transforms.functional import gaussian_blur

def readFileList(file_path):
    filelist = open(file_path).read().splitlines()
    return filelist

def readAllFilesFromFolder(file_path, extension):
    filelist = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.split(".")[1] == extension]
    return filelist

def binarize_saliency(saliency):
    threshold = threshold_otsu(saliency)
    bin_sal = np.copy(saliency)
    bin_sal[saliency>threshold] = 1
    bin_sal[saliency<=threshold] = 0
    
    return bin_sal

def get_files_extension(folder):
    ext = None
    for f in os.listdir(folder):
        if os.path.isfile(folder+"/"+f):
            ext = f.split(".")[-1]
    assert ext != None, "No files found in "+folder
    return ext
        
        
#------------------- MORPHOLOGY UTILITIES -------------------

def erosion(x, kernel_size=5):

    return -F.max_pool2d(
        -x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size//2
    )


def reconstruction_by_erosion(marker, mask, kernel_size=5, check_every=10):
    """
    Inferior reconstruction.

    marker >= mask
    shape: [B,C,H,W]
    """
    max_iters = sum(marker.shape[-2:])
    current = torch.maximum(marker, mask) 

    for it in range(max_iters):
        next_img = torch.maximum(erosion(current, kernel_size), mask)

        if(it + 1) % check_every == 0 and torch.equal(next_img, current):
                return next_img
        
        current = next_img

    return current

def combine_inferior_reconstruction(saliency_list, kernel_size=5, deep_blur_sigma=0.0):
    """
    saliency_list: ordered [shallow=index 0, ..., deep=index -1]
    Each: [B, 1, H, W], values in [0, 1]

    For each pair (L, L+K), replaces the deeper map with
    inferior_rec(marker=shallow, mask=min(shallow, deep)).

    deep_blur_sigma > 0 Gaussian-blurs the deep map before it becomes the
    erosion floor. The deep map is a heavily downsampled grid upsampled back to
    full size, so it is blocky at the coarse-grid resolution; the raw min(shallow,
    deep) floor snaps the reconstructed boundary to those blocks. Blurring turns
    that floor into a smooth ramp so the boundary follows a smooth curve instead
    of the grid facets. Trades a little of the deep map's gating confidence for
    smoother borders; sigma=0 keeps the original hard floor.
    """
    maps = [s.clone() for s in saliency_list]

    for i in range(len(maps) - 1):
        shallow = maps[i]        # layer L
        deep    = maps[i + 1]    # layer L+K

        if deep_blur_sigma > 0:
            # cover ±3 sigma, kernel size forced odd
            ksize = int(2 * round(3 * deep_blur_sigma) + 1)
            deep = gaussian_blur(deep, kernel_size=ksize, sigma=deep_blur_sigma)

        mask   = torch.minimum(shallow, deep)   # enforces marker >= mask
        marker = shallow

        maps[i + 1] = reconstruction_by_erosion(marker, mask, kernel_size)

    return maps[-1]


def dilation(x, kernel_size=5):

    return F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size//2
    )


def reconstruction_by_dilation(marker, mask, kernel_size=5, check_every=10):
    """
    Superior reconstruction (dual of reconstruction_by_erosion).

    marker <= mask
    shape: [B,C,H,W]
    """
    max_iters = sum(marker.shape[-2:])
    current = torch.minimum(marker, mask)

    for it in range(max_iters):
        next_img = torch.minimum(dilation(current, kernel_size), mask)

        if(it + 1) % check_every == 0 and torch.equal(next_img, current):
                return next_img

        current = next_img

    return current


def combine_superior_reconstruction(saliency_list, kernel_size=5):
    """
    saliency_list: ordered [shallow=index 0, ..., deep=index -1]
    Each: [B, 1, H, W], values in [0, 1]

    Dual of combine_inferior_reconstruction. Instead of eroding the shallow map
    down to a deep-gated floor (which shrinks foreground and snaps borders to the
    coarse deep grid), it GROWS the deep (semantic, confident) seed up to the
    shallow (high-res, sharp) mask via reconstruction by dilation. Processed
    deepest -> shallowest, so the final result lives at the shallowest map's
    resolution: localization is seeded by the deep map while the borders are
    clamped to the data-driven shallow shape (no square-SE staircase, recovers
    the recall that erosion/gating discard).
    """
    maps = [s.clone() for s in saliency_list]

    result = maps[-1]                           # deepest = confident seed
    for i in range(len(maps) - 2, -1, -1):
        mask   = maps[i]                        # higher-res (sharper) shape
        marker = torch.minimum(result, mask)    # enforces marker <= mask

        result = reconstruction_by_dilation(marker, mask, kernel_size)

    return result                               # shallowest resolution -> sharp
