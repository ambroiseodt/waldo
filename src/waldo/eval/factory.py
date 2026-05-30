r"""
Linear probing instantiation factory.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import logging
from typing import Any

from ..infra import build_with_type_check
from .linear import LinearProbing, LinearProbingConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#   Build linear probing
# ------------------------------------------------------------------------------
def build_linear_probing(
    config: dict[str, Any],
) -> LinearProbing:
    r"""
    Build linear probing.

    Parameters
    ----------
    config: dict
        Dictionary containing the configuration details.

    Returns
    -------
    linear_probing: LinearProbing
        An instance of the specified linear probing instance.
    """

    config_obj = build_with_type_check(LinearProbingConfig, config)
    linear_probing = LinearProbing(config_obj)
    return linear_probing
