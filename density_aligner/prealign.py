"""
Pre-alignment utilities.

Optional tools for aligning molecules to their principal axes before
running the main density alignment. This can help when you want to
set up consistent orientations, e.g., for generating cube files from
quantum chemistry calculations.
"""

import numpy as np
from typing import List, Tuple, Optional
from .cube_io import Atom, CubeData


# Atomic masses for center of mass calculation
ATOMIC_MASSES = {
    1: 1.008, 2: 4.003, 3: 6.941, 4: 9.012, 5: 10.81, 6: 12.01,
    7: 14.01, 8: 16.00, 9: 19.00, 10: 20.18, 11: 22.99, 12: 24.31,
    13: 26.98, 14: 28.09, 15: 30.97, 16: 32.07, 17: 35.45, 18: 39.95,
    19: 39.10, 20: 40.08, 35: 79.90, 53: 126.9
}


def get_atomic_mass(atomic_num: int) -> float:
    """Get atomic mass for an element."""
    return ATOMIC_MASSES.get(atomic_num, 12.0)  # Default to carbon


def calculate_center_of_mass(atoms: List[Atom]) -> np.ndarray:
    """
    Calculate center of mass of atoms.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.

    Returns
    -------
    np.ndarray
        Center of mass coordinates.
    """
    total_mass = 0.0
    weighted_coords = np.zeros(3)

    for atom in atoms:
        mass = get_atomic_mass(atom.atomic_num)
        total_mass += mass
        weighted_coords += mass * atom.coords

    return weighted_coords / total_mass


def calculate_inertia_tensor(atoms: List[Atom], center: np.ndarray) -> np.ndarray:
    """
    Calculate inertia tensor around center.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.
    center : np.ndarray
        Center point for inertia calculation.

    Returns
    -------
    np.ndarray
        3x3 inertia tensor.
    """
    I = np.zeros((3, 3))

    for atom in atoms:
        mass = get_atomic_mass(atom.atomic_num)
        r = atom.coords - center

        # Diagonal elements
        I[0, 0] += mass * (r[1]**2 + r[2]**2)
        I[1, 1] += mass * (r[0]**2 + r[2]**2)
        I[2, 2] += mass * (r[0]**2 + r[1]**2)

        # Off-diagonal elements
        I[0, 1] -= mass * r[0] * r[1]
        I[0, 2] -= mass * r[0] * r[2]
        I[1, 2] -= mass * r[1] * r[2]

    # Symmetric
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    return I


def principal_axes_alignment(atoms: List[Atom]) -> Tuple[List[Atom], np.ndarray, np.ndarray]:
    """
    Align atoms to principal axes of inertia.

    Translates center of mass to origin and rotates to align with
    principal axes.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.

    Returns
    -------
    tuple
        (aligned_atoms, rotation_matrix, center_of_mass)
    """
    # Calculate center of mass
    com = calculate_center_of_mass(atoms)

    # Calculate inertia tensor
    I = calculate_inertia_tensor(atoms, com)

    # Get principal axes (eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    # Sort by eigenvalue (smallest first for most extended axis)
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] = -eigenvectors[:, 2]

    # Rotation matrix (eigenvectors as columns)
    R = eigenvectors.T

    # Transform atoms
    aligned_atoms = []
    for atom in atoms:
        # Translate to origin and rotate
        coords_centered = atom.coords - com
        coords_rotated = R @ coords_centered
        aligned_atoms.append(Atom(atom.atomic_num, atom.charge, coords_rotated))

    return aligned_atoms, R, com


def prealign_molecule(cube_data: CubeData) -> Tuple[CubeData, np.ndarray, np.ndarray]:
    """
    Pre-align molecule in cube file to principal axes.

    Note: This only aligns atoms. The density grid is not transformed.
    For density alignment, use the main alignment functions.

    Parameters
    ----------
    cube_data : CubeData
        Input cube data.

    Returns
    -------
    tuple
        (aligned_cube_data, rotation_matrix, original_center_of_mass)
    """
    aligned_atoms, R, com = principal_axes_alignment(cube_data.atoms)

    # Create new CubeData with aligned atoms
    # Note: density is not transformed
    aligned_cube = CubeData(
        title1=cube_data.title1,
        title2=cube_data.title2,
        atoms=aligned_atoms,
        origin=cube_data.origin,
        voxel_vectors=cube_data.voxel_vectors,
        grid_shape=cube_data.grid_shape,
        density=cube_data.density
    )

    return aligned_cube, R, com


def calculate_rmsd(atoms1: List[Atom], atoms2: List[Atom]) -> float:
    """
    Calculate RMSD between two sets of atoms.

    Parameters
    ----------
    atoms1, atoms2 : List[Atom]
        Lists of atoms (must have same length).

    Returns
    -------
    float
        RMSD in same units as coordinates.
    """
    if len(atoms1) != len(atoms2):
        raise ValueError("Atom lists must have same length")

    coords1 = np.array([a.coords for a in atoms1])
    coords2 = np.array([a.coords for a in atoms2])

    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def get_bounding_box(atoms: List[Atom], padding: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box for atoms with padding.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.
    padding : float
        Padding to add around atoms (in Bohr).

    Returns
    -------
    tuple
        (min_coords, max_coords)
    """
    coords = np.array([a.coords for a in atoms])
    min_coords = np.min(coords, axis=0) - padding
    max_coords = np.max(coords, axis=0) + padding
    return min_coords, max_coords
