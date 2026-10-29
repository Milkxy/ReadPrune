"""ReadPrune: response-guided selection of unchanged visual tokens.

The NumPy/SciPy selection API does not load model libraries. Import
``readprune.readout.Readout`` explicitly when PyTorch readout support is needed.
"""

from .selection import select_tokens

__all__ = ["select_tokens"]
__version__ = "0.1.0"
