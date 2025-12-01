"""
Density Aligner - Molecular alignment using electron density overlap.

A tool for aligning molecules based on their quantum mechanical electron
density distributions from cube files.
"""

__version__ = "1.0.0"
__author__ = "Density Aligner Contributors"

from .cube_io import read_cube, write_cube
from .spatial_aligner import SpatialAligner, align_densities
from .prealign import prealign_molecule, principal_axes_alignment

__all__ = [
    "read_cube",
    "write_cube",
    "SpatialAligner",
    "align_densities",
    "prealign_molecule",
    "principal_axes_alignment",
]
