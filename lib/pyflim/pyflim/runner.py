import os
import numpy as np
from torchvision import utils
from torch.utils.data import DataLoader
from torchvision import transforms as F
from tqdm import tqdm

from pyflim.flim import FLIMModel
from pyflim.datasets.base import FLIMBaseDataset


class FLIMRunner:
    def __init__(self, model: FLIMModel, device: str = "cpu"):
        self.model = model
        self.device = device

    def run(
        self,
        dataset: FLIMBaseDataset | DataLoader,
        output_folder: str,
        batch_size: int = 1,
        decoder_block: list = None,
        save_activations: bool = False,
        save_activations_only: bool = False,
    ):
        print(f"\n{'='*60}")
        print(f"Starting FLIMRunner inference")
        print(f"Output folder: {output_folder}")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"Save activations: {save_activations}")
        print(f"Save activations only: {save_activations_only}")
        print(f"{'='*60}\n")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"✓ Created output folder: {output_folder}")

        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, batch_size=batch_size, drop_last=False)
            print(f"✓ Created DataLoader with batch_size={batch_size}")

        # Resolve which blocks will be decoded.
        # ``decoder_block`` argument takes precedence; otherwise use the YAML
        # value coming from ``decoder_layers`` (with fallback to ``inputs``).
        if decoder_block is not None:
            active_decoder_blocks = decoder_block
        elif self.model.architecture.decoder is not None:
            active_decoder_blocks = self.model.architecture.decoder.get_decoder_layers()
        else:
            active_decoder_blocks = []

        # Create one output sub-folder per decoded block.
        for block in active_decoder_blocks:
            os.makedirs(
                os.path.join(output_folder, f"block_{block}"), exist_ok=True
            )

        if save_activations or save_activations_only:
            for block in active_decoder_blocks:
                os.makedirs(
                    os.path.join(output_folder, f"block_{block}"), exist_ok=True
                )

        for sample_batch in tqdm(dataset, desc="Processing batches", leave=True):
            X = sample_batch["image"].to(self.device)

            if not save_activations:
                y_hat = self.model.forward(
                    X,
                    decoder_block=active_decoder_blocks,
                    return_activations=False,
                )
            else:
                y_hat, y_activations = self.model.forward(
                    X,
                    decoder_block=active_decoder_blocks,
                    return_activations=True,
                )

            original_sizes = sample_batch["original_size"]
            image_paths = sample_batch["image_path"]

            for block, y_ in zip(active_decoder_blocks, y_hat):
                for image_idx, (image_path, y, original_size) in enumerate(
                    zip(image_paths, y_, original_sizes)
                ):
                    out_size = y.shape[-2:]
                    if (
                        out_size[0] != original_size[0]
                        or out_size[1] != original_size[1]
                    ):
                        y = F.interpolate(
                            y.unsqueeze(0),
                            [original_size[0], original_size[1]],
                            mode="bilinear",
                            align_corners=True,
                        ).squeeze(0)

                    saliency = y[0].detach()

                    if not save_activations_only:
                        self._save_saliency(
                            saliency,
                            output_folder,
                            image_path.split("/")[-1],
                            block,
                        )

            for i, image_path in enumerate(image_paths):
                if save_activations:
                    image_file = image_path.split("/")[-1]
                    output_path = os.path.join(
                        output_folder, image_file.replace(".png", ".npz")
                    )
                    block_data = {
                        f"block_{block+1}": y.numpy()
                        for block, y in zip(active_decoder_blocks, y_activations[i])
                    }
                    self._save_activations(block_data, output_path)

                    del saliency

            del y_hat
            del X
            print()

        print(f"{'='*60}")
        print(f"✓ Inference complete! Results saved to: {output_folder}")
        print(f"{'='*60}\n")

    def _save_saliency(
        self, saliency: np.ndarray, output_folder: str, image_path: str, block: int
    ) -> None:
        utils.save_image(
            saliency,
            os.path.join(output_folder, f"block_{block}", image_path),
        )

    def _save_activations(self, block_data: dict, output_path: str) -> None:
        np.savez_compressed(output_path, **block_data)