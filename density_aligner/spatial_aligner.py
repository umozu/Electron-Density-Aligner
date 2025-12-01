"""
Spatial alignment using electron density overlap.

This module handles the core alignment algorithm. It takes two electron
density grids (from cube files) and finds the rotation + translation that
best overlaps them. We use PyTorch for automatic differentiation so we can
do gradient descent on the overlap metric.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from .cube_io import CubeData, Atom


@dataclass
class AlignmentResult:
    """Container for alignment results."""
    quaternion: np.ndarray
    translation: np.ndarray
    spatial_overlap: float
    true_binary_overlap: float
    iterations: int
    pre_rotation: str
    seed: int


class SpatialAligner(nn.Module):
    """
    The main alignment engine.

    This is a PyTorch module that holds the transformation parameters
    (rotation as a quaternion, translation as a 3D vector) and can
    transform a density grid. By making it a nn.Module, we get automatic
    gradient computation for free.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize aligner with optional random seed.

        Parameters
        ----------
        random_seed : int, optional
            Seed for reproducible random initialization.
        """
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)
            q = torch.randn(4)
            q = q / torch.norm(q)
            self.rotation_quaternion = nn.Parameter(q)
            self.translation = nn.Parameter(torch.randn(3) * 0.5)
        else:
            self.rotation_quaternion = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            self.translation = nn.Parameter(torch.zeros(3))

    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to 3x3 rotation matrix."""
        q = F.normalize(q, dim=-1)
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
        ])

    def transform_density(self, density: torch.Tensor, origin: np.ndarray,
                          voxel_vectors: Tuple[np.ndarray, np.ndarray, np.ndarray],
                          grid_center: np.ndarray) -> torch.Tensor:
        """
        Transform density grid using current rotation and translation.

        Parameters
        ----------
        density : torch.Tensor
            3D density grid.
        origin : np.ndarray
            Grid origin coordinates.
        voxel_vectors : tuple
            Three voxel vectors defining the grid.
        grid_center : np.ndarray
            Center of the grid for rotation.

        Returns
        -------
        torch.Tensor
            Transformed density grid.
        """
        nx, ny, nz = density.shape
        device = density.device
        rotation_matrix = self.quaternion_to_rotation_matrix(self.rotation_quaternion)

        # Create grid indices
        i_idx = torch.arange(nx, device=device, dtype=torch.float32)
        j_idx = torch.arange(ny, device=device, dtype=torch.float32)
        k_idx = torch.arange(nz, device=device, dtype=torch.float32)
        ii, jj, kk = torch.meshgrid(i_idx, j_idx, k_idx, indexing='ij')

        # Convert to tensors
        voxel_x = torch.tensor(voxel_vectors[0], device=device, dtype=torch.float32)
        voxel_y = torch.tensor(voxel_vectors[1], device=device, dtype=torch.float32)
        voxel_z = torch.tensor(voxel_vectors[2], device=device, dtype=torch.float32)
        origin_tensor = torch.tensor(origin, device=device, dtype=torch.float32)
        center_tensor = torch.tensor(grid_center, device=device, dtype=torch.float32)

        # Calculate physical coordinates
        coords = (ii.unsqueeze(-1) * voxel_x.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
                  jj.unsqueeze(-1) * voxel_y.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
                  kk.unsqueeze(-1) * voxel_z.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
                  origin_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        # Center, rotate, translate
        coords_centered = coords - center_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        coords_flat = coords_centered.reshape(-1, 3)

        inv_rotation = rotation_matrix.T
        rotated_coords_flat = torch.matmul(coords_flat, inv_rotation.T)
        rotated_coords_centered = rotated_coords_flat.reshape(nx, ny, nz, 3)

        rotated_coords = (rotated_coords_centered -
                          self.translation.unsqueeze(0).unsqueeze(0).unsqueeze(0) +
                          center_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0))

        # Convert back to grid indices
        rotated_coords_shifted = rotated_coords - origin_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        voxel_matrix = torch.stack([voxel_x, voxel_y, voxel_z], dim=1)
        inv_voxel_matrix = torch.linalg.inv(voxel_matrix)

        indices_flat = torch.matmul(rotated_coords_shifted.reshape(-1, 3), inv_voxel_matrix.T)
        indices = indices_flat.reshape(nx, ny, nz, 3)

        # Normalize for grid_sample
        indices_norm = 2.0 * indices / torch.tensor([nx-1, ny-1, nz-1],
                                                     device=device, dtype=torch.float32) - 1.0
        indices_norm = indices_norm.flip(dims=[3])

        # Sample density
        density_reshaped = density.unsqueeze(0).unsqueeze(0)
        indices_grid = indices_norm.unsqueeze(0)

        output = F.grid_sample(
            density_reshaped,
            indices_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return output.squeeze()

    def calculate_spatial_overlap(self, density1: torch.Tensor,
                                   density2: torch.Tensor,
                                   threshold: float = 1e-6) -> torch.Tensor:
        """
        Calculate differentiable spatial overlap using soft thresholding.

        Parameters
        ----------
        density1 : torch.Tensor
            Reference density.
        density2 : torch.Tensor
            Query density.
        threshold : float
            Density threshold for occupancy.

        Returns
        -------
        torch.Tensor
            Spatial overlap coefficient (0-1).
        """
        d1 = torch.abs(density1)
        d2 = torch.abs(density2)

        # Soft sigmoid thresholding for differentiability
        scale = 1000.0
        soft1 = torch.sigmoid((d1 - threshold) * scale)
        soft2 = torch.sigmoid((d2 - threshold) * scale)

        # Soft AND and OR operations
        both_soft = soft1 * soft2
        either_soft = soft1 + soft2 - both_soft

        spatial_overlap = torch.sum(both_soft) / (torch.sum(either_soft) + 1e-10)
        return spatial_overlap

    def forward(self, ligand_density: torch.Tensor, reference_density: torch.Tensor,
                origin: np.ndarray,
                voxel_vectors: Tuple[np.ndarray, np.ndarray, np.ndarray],
                grid_center: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: transform ligand and compute overlap.

        Returns
        -------
        tuple
            (loss, spatial_overlap, transformed_density)
        """
        with torch.no_grad():
            self.translation.data = torch.clamp(self.translation.data, -10.0, 10.0)

        transformed = self.transform_density(ligand_density, origin, voxel_vectors, grid_center)
        spatial = self.calculate_spatial_overlap(transformed, reference_density)
        loss = -spatial

        return loss, spatial, transformed


def rotate_density_180(density: np.ndarray, axis: str) -> np.ndarray:
    """
    Rotate density grid 180 degrees around specified axis.

    Parameters
    ----------
    density : np.ndarray
        3D density array.
    axis : str
        Rotation axis: 'x', 'y', or 'xy'.

    Returns
    -------
    np.ndarray
        Rotated density array.
    """
    if axis == 'x':
        return density[:, ::-1, ::-1].copy()
    elif axis == 'y':
        return density[::-1, :, ::-1].copy()
    elif axis == 'xy':
        temp = density[:, ::-1, ::-1]
        return temp[::-1, :, ::-1].copy()
    return density.copy()


def rotate_atoms_180(atoms: List[Atom], axis: str, center: np.ndarray) -> List[Atom]:
    """
    Rotate atoms 180 degrees around specified axis.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.
    axis : str
        Rotation axis: 'x', 'y', or 'xy'.
    center : np.ndarray
        Center of rotation.

    Returns
    -------
    List[Atom]
        Rotated atoms.
    """
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    elif axis == 'y':
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    elif axis == 'xy':
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        R = np.eye(3)

    rotated = []
    for atom in atoms:
        coords_centered = atom.coords - center
        coords_rotated = R @ coords_centered
        coords_final = coords_rotated + center
        rotated.append(Atom(atom.atomic_num, atom.charge, coords_final))
    return rotated


def quaternion_to_rotation_matrix_numpy(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix (numpy version)."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def transform_atoms(atoms: List[Atom], quaternion: np.ndarray,
                    translation: np.ndarray, center: np.ndarray) -> List[Atom]:
    """
    Apply rotation and translation to atoms.

    Parameters
    ----------
    atoms : List[Atom]
        List of atoms.
    quaternion : np.ndarray
        Rotation quaternion [w, x, y, z].
    translation : np.ndarray
        Translation vector.
    center : np.ndarray
        Center of rotation.

    Returns
    -------
    List[Atom]
        Transformed atoms.
    """
    R = quaternion_to_rotation_matrix_numpy(quaternion)

    transformed = []
    for atom in atoms:
        coords_centered = atom.coords - center
        coords_rotated = R @ coords_centered
        coords_final = coords_rotated + translation + center
        transformed.append(Atom(atom.atomic_num, atom.charge, coords_final))
    return transformed


def calculate_true_binary_overlap(ref_density: np.ndarray,
                                   lig_density: np.ndarray,
                                   threshold: float = 1e-6) -> float:
    """
    Calculate true binary spatial overlap.

    Parameters
    ----------
    ref_density : np.ndarray
        Reference density.
    lig_density : np.ndarray
        Ligand density.
    threshold : float
        Density threshold.

    Returns
    -------
    float
        Binary spatial overlap (0-1).
    """
    ref_nonzero = np.abs(ref_density) > threshold
    lig_nonzero = np.abs(lig_density) > threshold

    both = np.sum(ref_nonzero & lig_nonzero)
    either = np.sum(ref_nonzero | lig_nonzero)

    return both / (either + 1e-10)


def run_single_alignment(ref_cube: CubeData, lig_density: np.ndarray,
                          seed: int, max_iters: int = 1000,
                          verbose: bool = False) -> Tuple[float, Dict, int]:
    """
    Run a single alignment optimization.

    Parameters
    ----------
    ref_cube : CubeData
        Reference cube data.
    lig_density : np.ndarray
        Ligand density (possibly pre-rotated).
    seed : int
        Random seed.
    max_iters : int
        Maximum iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple
        (best_spatial, best_params, iterations)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ref_tensor = torch.tensor(ref_cube.density, dtype=torch.float32, device=device)
    lig_tensor = torch.tensor(lig_density, dtype=torch.float32, device=device)

    model = SpatialAligner(random_seed=seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, min_lr=1e-5)

    best_spatial = -float('inf')
    best_params = None
    spatial_history = []

    for iteration in range(max_iters):
        optimizer.zero_grad()

        loss, spatial, _ = model(
            lig_tensor, ref_tensor,
            ref_cube.origin, ref_cube.voxel_vectors, ref_cube.grid_center
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        current_spatial = spatial.item()
        spatial_history.append(current_spatial)
        scheduler.step(current_spatial)

        if current_spatial > best_spatial:
            best_spatial = current_spatial
            best_params = {
                'quaternion': model.rotation_quaternion.detach().cpu().numpy().copy(),
                'translation': model.translation.detach().cpu().numpy().copy(),
                'iteration': iteration
            }

        # Early stopping check
        if iteration >= 30 and iteration % 10 == 0:
            if len(spatial_history) > 10:
                recent_best = max(spatial_history[-10:])
                previous_best = max(spatial_history[:-10]) if len(spatial_history) > 10 else 0
                if previous_best > 0:
                    improvement = (recent_best - previous_best) / previous_best
                    if improvement < 0.0001:
                        break

        if verbose and iteration % 50 == 0:
            print(f"  Iter {iteration}: spatial={current_spatial:.4f}")

    return best_spatial, best_params, iteration + 1


def align_densities(ref_cube: CubeData, lig_cube: CubeData,
                    n_starts: int = 10,
                    seeds: Optional[List[int]] = None,
                    max_iters: int = 1000,
                    verbose: bool = True) -> AlignmentResult:
    """
    Align ligand density to reference density.

    Uses multi-start optimization with 4 pre-rotation positions and
    multiple random seeds to find the best alignment.

    Parameters
    ----------
    ref_cube : CubeData
        Reference molecule cube data.
    lig_cube : CubeData
        Ligand molecule cube data.
    n_starts : int
        Number of random seeds per rotation.
    seeds : List[int], optional
        Specific seeds to use. If None, uses default seeds.
    max_iters : int
        Maximum iterations per optimization.
    verbose : bool
        Print progress.

    Returns
    -------
    AlignmentResult
        Best alignment result.
    """
    if seeds is None:
        seeds = [42, 55, 78, 123, 200, 314, 500, 789, 911, 999][:n_starts]

    # Pre-rotation positions
    positions = {
        'original': lig_cube.density.copy(),
        '180_x': rotate_density_180(lig_cube.density, 'x'),
        '180_y': rotate_density_180(lig_cube.density, 'y'),
        '180_xy': rotate_density_180(lig_cube.density, 'xy')
    }

    best_result = None
    best_spatial = -float('inf')
    total_runs = len(positions) * len(seeds)
    run_count = 0

    if verbose:
        print(f"Running {total_runs} alignment optimizations...")
        print("-" * 50)

    for pos_name, lig_density in positions.items():
        for seed in seeds:
            run_count += 1

            spatial, params, iters = run_single_alignment(
                ref_cube, lig_density, seed, max_iters, verbose=False
            )

            if verbose:
                print(f"  [{run_count}/{total_runs}] {pos_name:8s} seed={seed:3d}: "
                      f"spatial={spatial:.4f} ({iters} iters)")

            if spatial > best_spatial:
                best_spatial = spatial
                best_result = {
                    'quaternion': params['quaternion'],
                    'translation': params['translation'],
                    'spatial': spatial,
                    'position': pos_name,
                    'seed': seed,
                    'iterations': iters
                }

    # Calculate true binary overlap for best result
    pos = best_result['position']
    if pos == '180_x':
        final_lig_density = rotate_density_180(lig_cube.density, 'x')
    elif pos == '180_y':
        final_lig_density = rotate_density_180(lig_cube.density, 'y')
    elif pos == '180_xy':
        final_lig_density = rotate_density_180(lig_cube.density, 'xy')
    else:
        final_lig_density = lig_cube.density.copy()

    # Transform density with best parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatialAligner()
    model.rotation_quaternion.data = torch.tensor(best_result['quaternion'], dtype=torch.float32)
    model.translation.data = torch.tensor(best_result['translation'], dtype=torch.float32)
    model = model.to(device)

    with torch.no_grad():
        lig_tensor = torch.tensor(final_lig_density, dtype=torch.float32, device=device)
        transformed = model.transform_density(
            lig_tensor, ref_cube.origin, ref_cube.voxel_vectors, ref_cube.grid_center
        ).cpu().numpy()

    true_binary = calculate_true_binary_overlap(ref_cube.density, transformed)

    if verbose:
        print("-" * 50)
        print(f"Best: {best_result['position']} seed={best_result['seed']} "
              f"spatial={best_result['spatial']:.4f} binary={true_binary:.4f}")

    return AlignmentResult(
        quaternion=best_result['quaternion'],
        translation=best_result['translation'],
        spatial_overlap=best_result['spatial'],
        true_binary_overlap=true_binary,
        iterations=best_result['iterations'],
        pre_rotation=best_result['position'],
        seed=best_result['seed']
    )
