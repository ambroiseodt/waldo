r"""
Model instantiation factory for VFMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import asdict
from typing import Any

import torch.nn as nn

from ..infra import build_with_type_check


def build_model(config: dict[str, Any], return_config: bool = False) -> nn.Module | tuple[nn.Module, dict]:
    r"""
    Initialize model based on the specified implementation given in the config file.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.
    return_config: bool, default=False
        Whether to return the model configuration.

    Returns
    -------
    model: nn.Module | tuple[nn.Module, dict]
        An instance of the specified model implementation.
        If return_config is True, the config file is also returned as a dictionary.
    """

    # Argument parsing
    model_name = config.pop("model_name", "vit")

    match model_name.lower():
        # Vision Foundation Models (VFMs)
        case "aimv2":
            from waldo.models import Aimv2, Aimv2Config

            model_type = Aimv2
            config_obj = build_with_type_check(Aimv2Config, config)

        case "dinov2":
            from waldo.models import Dinov2, Dinov2Config

            model_type = Dinov2
            config_obj = build_with_type_check(Dinov2Config, config)

        case "dinov3":
            from waldo.models import Dinov3, Dinov3Config

            model_type = Dinov3
            config_obj = build_with_type_check(Dinov3Config, config)

        case "vit":
            from waldo.models import ViT, ViTConfig

            model_type = ViT
            config_obj = build_with_type_check(ViTConfig, config)

        # Large Language Models (LLMs)
        case "gemma3":
            from waldo.models import Gemma3, Gemma3Config

            model_type = Gemma3
            config_obj = build_with_type_check(Gemma3Config, config)

        case "llama3":
            from waldo.models import Llama3, Llama3Config

            model_type = Llama3
            config_obj = build_with_type_check(Llama3Config, config)

        case "qwen3":
            from waldo.models import Qwen3, Qwen3Config

            model_type = Qwen3
            config_obj = build_with_type_check(Qwen3Config, config)

        case _:
            raise ValueError(f"Model {model_name} not found.")

    # Initialize the model
    model = model_type(config_obj)

    # Return model and config
    if return_config:
        return model, asdict(config_obj)

    return model
