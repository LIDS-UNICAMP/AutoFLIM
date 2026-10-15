import yaml
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearnerConfig:
    type: str  # "classical", "bofp", "multiclass_bofp", null
    clusterer: str 
    clusterer_kwargs: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)

@dataclass
class BlockConfig:
    name: str
    block_type: str
    inputs: list[str]
    conv_type: str = "conv"
    noutput_channels: int = 32
    kernel_size: list[int] = field(default_factory=lambda: [3, 3])
    nkernels_per_marker: int = 10
    dilation_values: list[int] | int | None = None
    stride: int = 1
    padding_mode: str = "reflect"
    activation: str = "relu"
    learnable: bool = True
    pooling: dict | None = None
    learner: LearnerConfig | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize dilation_values to always be a list [h, w]"""
        if self.dilation_values is None:
            self.dilation_values = [1, 1]
        elif isinstance(self.dilation_values, int):
            self.dilation_values = [self.dilation_values, self.dilation_values]

@dataclass
class HeadConfig:
        """
        Configuration for an optional post-decoder head.
    
        The head receives the output of the adaptive decoder for each decoded layer
        and further processes it (e.g., IFT-based delineation, learned refinement).
    
        Attributes:
            type:   Head name as registered in HeadsRegistry (e.g. "ift_segmentation").
            kwargs: Arbitrary keyword arguments forwarded to the head constructor.
    """
        type: str
        kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class DecoderConfig:
    """
        Configuration for the adaptive decoder stage.
    
        Attributes:
            inputs:           Encoder block names whose feature maps are combined as
                            inputs to the decoder (for future multi-input fusion).
                            **Not** the list of blocks to decode — use decoder_layers
                            for that. Left unchanged from the original semantics.
            decoder_layers:   Encoder block names whose outputs will be decoded
                            (i.e. passed through the adaptive decoder and saved).
                            If None or empty, falls back to ``inputs`` for backwards
                            compatibility.
            combine_strategy: How to fuse multiple decoded outputs ("mean", "add",
                            or None to keep them separate).
            type:             Decoder implementation key in DecoderRegistry.
            adaptation_fn:    Adaptation function name used inside the decoder.
            deep_blur_sigma:  Gaussian blur sigma applied to the deep map before it
                            becomes the erosion floor in "inferior_reconstruction"
                            fusion (softens the coarse deep-grid boundary). 0 = no
                            blur (hard floor). Defaults to 0.0 for backward
                            compatibility with configs that predate this field.
            kwargs:           Extra keyword arguments forwarded to the decoder.
    """
    inputs: list[str]
    combine_strategy: str | None
    decoder_layers: list[str] | None = None
    type: str = "vanilla_adaptive_decoder"
    adaptation_fn: str = "robust_weights"
    deep_blur_sigma: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=dict)
    multiclass_combine: str | None = None  # e.g. "argmax" for multiclass saliency maps

    def get_decoder_layers(self) -> list[str]:
        """
            Return the list of blocks that should be decoded.
    
            If ``decoder_layers`` is explicitly set, use it; otherwise fall back to
            ``inputs`` for backwards compatibility with configs that predate this field.
        """
        return self.decoder_layers if self.decoder_layers else self.inputs
    

@dataclass
class ArchitectureConfig:
    name: str
    blocks: list[BlockConfig]  # ordered list of all nodes
    decoder: DecoderConfig | None = None
    stdev_factor: float = 1e-8
    network_type: str = "regular"
    use_bias: bool = False
    use_marker_based_normalization: bool = True
    ninput_channels: int = 3
    head: HeadConfig | None = None
    
    @staticmethod
    def from_yaml(path: str) -> "ArchitectureConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        # 1. parsing encoder configs
        blocks = []
        for b in data["blocks"]:
            learner_data = b.pop("learner", None)
            learner = (
                LearnerConfig(**learner_data)
                if isinstance(learner_data, dict)
                else None
            )

            blocks.append(BlockConfig(**b, learner=learner))
        data["blocks"] = blocks

        # 2. parsing decoder configs
        decoder_data = data.pop("decoder", None)
        _no_decoder = (
            decoder_data is None
            or decoder_data.get("type") in (None, "None", "null", "none")
        )
        if _no_decoder:
            data["decoder"] = None
        else:
            data["decoder"] = DecoderConfig(
                inputs=decoder_data.pop("inputs", []),
                decoder_layers=decoder_data.pop("decoder_layers", []),
                combine_strategy=decoder_data.pop("combine_strategy", None),
                type=decoder_data.pop("type", "vanilla_adaptive_decoder"),
                adaptation_fn=decoder_data.pop("adaptation_fn", "robust_weights"),
                deep_blur_sigma=decoder_data.pop("deep_blur_sigma", 0.0),
                kwargs=decoder_data.pop("kwargs", {})
            )

        # 3. Parse optional head config
        head_data = data.pop("head", None)
        data["head"] = HeadConfig(**head_data) if head_data else None

        return ArchitectureConfig(**data)

    def to_yaml(self, path: str) -> None:
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
