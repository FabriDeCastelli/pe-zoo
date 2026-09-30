"""Cached supra-Laplacian positional encodings of a sliding window of snapshots."""
from .encodings import PositionalEncodings, encode_windows

__all__ = ["encode_windows", "PositionalEncodings"]
