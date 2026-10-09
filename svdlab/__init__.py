"""svdlab - image compression with the singular value decomposition."""
from .core import (ChannelSVD, break_even_rank, compress, decompose, eckart_young_error,
                   energy, randomized_svd, rank_for_energy, reconstruct, storage_numbers,
                   storage_ratio, to_float, to_uint8)

__version__ = "1.0.0"
__all__ = [
    "ChannelSVD", "break_even_rank", "compress", "decompose", "eckart_young_error", "energy",
    "randomized_svd", "rank_for_energy", "reconstruct", "storage_numbers", "storage_ratio",
    "to_float", "to_uint8",
]
