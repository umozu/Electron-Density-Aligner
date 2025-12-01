"""
Command-line interface for density alignment.
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Optional

from .cube_io import read_cube, write_cube, write_xyz, write_combined_xyz, CubeData, Atom
from .spatial_aligner import (
    align_densities, AlignmentResult, rotate_density_180, rotate_atoms_180,
    transform_atoms, SpatialAligner, calculate_true_binary_overlap
)
import torch
import torch.nn.functional as F


def transform_density_with_params(density: np.ndarray, quaternion: np.ndarray,
                                   translation: np.ndarray, origin: np.ndarray,
                                   voxel_vectors: tuple, grid_center: np.ndarray) -> np.ndarray:
    """Transform density array with given parameters."""
    device = 'cpu'

    density_t = torch.tensor(density, dtype=torch.float32)

    q = torch.tensor(quaternion, dtype=torch.float32)
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[0], q[1], q[2], q[3]
    rotation_matrix = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)]),
        torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)]),
        torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    ])

    trans = torch.tensor(translation, dtype=torch.float32)

    nx, ny, nz = density.shape
    i_idx = torch.arange(nx, dtype=torch.float32)
    j_idx = torch.arange(ny, dtype=torch.float32)
    k_idx = torch.arange(nz, dtype=torch.float32)
    ii, jj, kk = torch.meshgrid(i_idx, j_idx, k_idx, indexing='ij')

    voxel_x = torch.tensor(voxel_vectors[0], dtype=torch.float32)
    voxel_y = torch.tensor(voxel_vectors[1], dtype=torch.float32)
    voxel_z = torch.tensor(voxel_vectors[2], dtype=torch.float32)
    origin_t = torch.tensor(origin, dtype=torch.float32)
    center_t = torch.tensor(grid_center, dtype=torch.float32)

    coords = (ii.unsqueeze(-1) * voxel_x + jj.unsqueeze(-1) * voxel_y +
              kk.unsqueeze(-1) * voxel_z + origin_t)

    coords_centered = coords - center_t
    coords_flat = coords_centered.reshape(-1, 3)

    inv_rotation = rotation_matrix.T
    rotated_coords_flat = torch.matmul(coords_flat, inv_rotation.T)
    rotated_coords_centered = rotated_coords_flat.reshape(nx, ny, nz, 3)

    rotated_coords = rotated_coords_centered - trans + center_t
    rotated_coords_shifted = rotated_coords - origin_t

    voxel_matrix = torch.stack([voxel_x, voxel_y, voxel_z], dim=1)
    inv_voxel_matrix = torch.linalg.inv(voxel_matrix)

    indices_flat = torch.matmul(rotated_coords_shifted.reshape(-1, 3), inv_voxel_matrix.T)
    indices = indices_flat.reshape(nx, ny, nz, 3)

    indices_norm = 2.0 * indices / torch.tensor([nx-1, ny-1, nz-1], dtype=torch.float32) - 1.0
    indices_norm = indices_norm.flip(dims=[3])

    density_reshaped = density_t.unsqueeze(0).unsqueeze(0)
    indices_grid = indices_norm.unsqueeze(0)

    output = F.grid_sample(density_reshaped, indices_grid, mode='bilinear',
                          padding_mode='zeros', align_corners=True)

    return output.squeeze().numpy()


def run_alignment(reference_cube: str, ligand_cube: str, output_dir: str,
                  n_starts: int = 10, max_iters: int = 1000,
                  verbose: bool = True) -> AlignmentResult:
    """
    Run density alignment and save results.

    Parameters
    ----------
    reference_cube : str
        Path to reference cube file.
    ligand_cube : str
        Path to ligand cube file.
    output_dir : str
        Output directory for results.
    n_starts : int
        Number of random starts per rotation.
    max_iters : int
        Maximum iterations per optimization.
    verbose : bool
        Print progress.

    Returns
    -------
    AlignmentResult
        Alignment results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load cube files
    if verbose:
        print("=" * 70)
        print("DENSITY ALIGNER")
        print("=" * 70)
        print(f"\nReference: {reference_cube}")
        print(f"Ligand:    {ligand_cube}")
        print(f"Output:    {output_dir}")

    ref_cube = read_cube(reference_cube)
    lig_cube = read_cube(ligand_cube)

    if verbose:
        print(f"\nReference: {ref_cube.n_atoms} atoms, grid {ref_cube.grid_shape}")
        print(f"Ligand:    {lig_cube.n_atoms} atoms, grid {lig_cube.grid_shape}")

    # Check grid compatibility
    if ref_cube.grid_shape != lig_cube.grid_shape:
        print(f"\nWarning: Grid shapes differ!")
        print(f"  Reference: {ref_cube.grid_shape}")
        print(f"  Ligand:    {lig_cube.grid_shape}")
        print("  Alignment may be suboptimal.")

    # Run alignment
    if verbose:
        print(f"\nStarting alignment ({4 * n_starts} optimizations)...")
        print("-" * 70)

    start_time = time.time()
    result = align_densities(ref_cube, lig_cube, n_starts=n_starts,
                             max_iters=max_iters, verbose=verbose)
    elapsed = time.time() - start_time

    if verbose:
        print("-" * 70)
        print(f"\nAlignment complete in {elapsed:.1f} seconds")
        print(f"\nBest result:")
        print(f"  Pre-rotation:     {result.pre_rotation}")
        print(f"  Seed:             {result.seed}")
        print(f"  Spatial overlap:  {result.spatial_overlap:.4f}")
        print(f"  Binary overlap:   {result.true_binary_overlap:.4f}")
        print(f"  Iterations:       {result.iterations}")

    # Apply transformation to get aligned atoms
    if result.pre_rotation == '180_x':
        lig_atoms = rotate_atoms_180(lig_cube.atoms, 'x', lig_cube.grid_center)
        lig_density = rotate_density_180(lig_cube.density, 'x')
    elif result.pre_rotation == '180_y':
        lig_atoms = rotate_atoms_180(lig_cube.atoms, 'y', lig_cube.grid_center)
        lig_density = rotate_density_180(lig_cube.density, 'y')
    elif result.pre_rotation == '180_xy':
        lig_atoms = rotate_atoms_180(lig_cube.atoms, 'xy', lig_cube.grid_center)
        lig_density = rotate_density_180(lig_cube.density, 'xy')
    else:
        lig_atoms = lig_cube.atoms
        lig_density = lig_cube.density.copy()

    # Apply optimized transformation
    aligned_atoms = transform_atoms(lig_atoms, result.quaternion,
                                     result.translation, ref_cube.grid_center)

    aligned_density = transform_density_with_params(
        lig_density, result.quaternion, result.translation,
        ref_cube.origin, ref_cube.voxel_vectors, ref_cube.grid_center
    )

    # Save outputs
    ref_name = os.path.splitext(os.path.basename(reference_cube))[0]
    lig_name = os.path.splitext(os.path.basename(ligand_cube))[0]

    # XYZ files
    ref_xyz = os.path.join(output_dir, f"{ref_name}_reference.xyz")
    aligned_xyz = os.path.join(output_dir, f"{lig_name}_aligned.xyz")
    combined_xyz = os.path.join(output_dir, "combined.xyz")

    write_xyz(ref_xyz, ref_cube.atoms, f"Reference: {ref_name}")
    write_xyz(aligned_xyz, aligned_atoms, f"Aligned: {lig_name}")
    write_combined_xyz(combined_xyz, ref_cube.atoms, aligned_atoms,
                       f"Reference ({ref_name}) + Aligned ({lig_name})")

    # Aligned cube file
    aligned_cube_path = os.path.join(output_dir, f"{lig_name}_aligned.cube")
    aligned_cube_data = CubeData(
        title1=f"Aligned {lig_name}",
        title2=f"Aligned to {ref_name} (overlap={result.true_binary_overlap:.4f})",
        atoms=aligned_atoms,
        origin=ref_cube.origin,
        voxel_vectors=ref_cube.voxel_vectors,
        grid_shape=ref_cube.grid_shape,
        density=aligned_density
    )
    write_cube(aligned_cube_path, aligned_cube_data)

    # Save alignment parameters
    params_file = os.path.join(output_dir, "alignment_params.npz")
    np.savez(params_file,
             quaternion=result.quaternion,
             translation=result.translation,
             pre_rotation=result.pre_rotation,
             seed=result.seed,
             spatial_overlap=result.spatial_overlap,
             binary_overlap=result.true_binary_overlap,
             iterations=result.iterations)

    # Save summary report
    report_file = os.path.join(output_dir, "alignment_report.txt")
    with open(report_file, 'w') as f:
        f.write("DENSITY ALIGNMENT REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Reference: {reference_cube}\n")
        f.write(f"Ligand:    {ligand_cube}\n\n")
        f.write(f"Reference atoms: {ref_cube.n_atoms}\n")
        f.write(f"Ligand atoms:    {lig_cube.n_atoms}\n")
        f.write(f"Grid shape:      {ref_cube.grid_shape}\n\n")
        f.write("ALIGNMENT RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Pre-rotation:    {result.pre_rotation}\n")
        f.write(f"Random seed:     {result.seed}\n")
        f.write(f"Iterations:      {result.iterations}\n")
        f.write(f"Spatial overlap: {result.spatial_overlap:.4f}\n")
        f.write(f"Binary overlap:  {result.true_binary_overlap:.4f}\n\n")
        f.write(f"Quaternion: [{result.quaternion[0]:.6f}, {result.quaternion[1]:.6f}, "
                f"{result.quaternion[2]:.6f}, {result.quaternion[3]:.6f}]\n")
        f.write(f"Translation: [{result.translation[0]:.6f}, {result.translation[1]:.6f}, "
                f"{result.translation[2]:.6f}] Bohr\n\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Reference XYZ:     {ref_xyz}\n")
        f.write(f"Aligned XYZ:       {aligned_xyz}\n")
        f.write(f"Combined XYZ:      {combined_xyz}\n")
        f.write(f"Aligned cube:      {aligned_cube_path}\n")
        f.write(f"Parameters:        {params_file}\n")

    if verbose:
        print(f"\nOutput files:")
        print(f"  {ref_xyz}")
        print(f"  {aligned_xyz}")
        print(f"  {combined_xyz}")
        print(f"  {aligned_cube_path}")
        print(f"  {params_file}")
        print(f"  {report_file}")

    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Align molecules using electron density overlap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s reference.cube ligand.cube -o output/
  %(prog)s ref.cube lig.cube -o results/ -n 5 --quiet

The alignment finds the optimal rotation and translation to maximize
the spatial overlap between electron density distributions.
        """
    )

    parser.add_argument("reference", help="Reference cube file")
    parser.add_argument("ligand", help="Ligand cube file to align")
    parser.add_argument("-o", "--output", default="alignment_output",
                        help="Output directory (default: alignment_output)")
    parser.add_argument("-n", "--n-starts", type=int, default=10,
                        help="Random starts per rotation (default: 10)")
    parser.add_argument("-m", "--max-iters", type=int, default=1000,
                        help="Max iterations per optimization (default: 1000)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.reference):
        print(f"Error: Reference file not found: {args.reference}")
        sys.exit(1)

    if not os.path.isfile(args.ligand):
        print(f"Error: Ligand file not found: {args.ligand}")
        sys.exit(1)

    # Run alignment
    try:
        result = run_alignment(
            args.reference,
            args.ligand,
            args.output,
            n_starts=args.n_starts,
            max_iters=args.max_iters,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n" + "=" * 70)
            print("ALIGNMENT COMPLETE")
            print(f"Binary spatial overlap: {result.true_binary_overlap:.4f}")
            print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
