#!/usr/bin/env python3
"""
Generate small test cube files for demonstration.

These are synthetic densities (Gaussian blobs), not real electron densities,
but they're useful for testing the alignment code without needing large
quantum chemistry calculations.

Run this script to create test files:
    python generate_test_cubes.py
"""

import numpy as np


def write_cube(filename, atoms, density, origin, voxel_size, title1="", title2=""):
    """Write a cube file."""
    nx, ny, nz = density.shape

    with open(filename, 'w') as f:
        f.write(f"{title1}\n")
        f.write(f"{title2}\n")

        # Number of atoms and origin
        f.write(f"{len(atoms):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")

        # Grid dimensions (voxel_size is in Bohr)
        f.write(f"{nx:5d} {voxel_size:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        f.write(f"{ny:5d} {0.0:12.6f} {voxel_size:12.6f} {0.0:12.6f}\n")
        f.write(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {voxel_size:12.6f}\n")

        # Atoms: atomic_number, charge, x, y, z
        for atom in atoms:
            f.write(f"{atom[0]:5d} {0.0:12.6f} {atom[1]:12.6f} {atom[2]:12.6f} {atom[3]:12.6f}\n")

        # Density values (6 per line)
        values = density.flatten()
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = " ".join(f"{v:12.5E}" for v in chunk)
            f.write(line + "\n")


def gaussian_blob(grid_shape, center, sigma, amplitude=1.0):
    """Create a 3D Gaussian blob on a grid."""
    nx, ny, nz = grid_shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    dist_sq = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2
    return amplitude * np.exp(-dist_sq / (2 * sigma**2))


def create_molecule_density(grid_shape, atom_positions, sigma=3.0):
    """Create a density as sum of Gaussian blobs at atom positions."""
    density = np.zeros(grid_shape)
    for pos in atom_positions:
        density += gaussian_blob(grid_shape, pos, sigma)
    return density


def main():
    # Grid parameters
    grid_size = 50  # Small grid for fast testing
    voxel_size = 0.5  # Bohr
    origin = np.array([-12.5, -12.5, -12.5])  # Center the grid around origin

    # Convert from Bohr coordinates to grid indices
    def bohr_to_grid(coords):
        return (coords - origin) / voxel_size

    # Define two simple "molecules" - just a few atoms each
    # Reference: a bent molecule (like water)
    ref_atoms_bohr = [
        [6, 0.0, 0.0, 0.0],      # C at origin
        [6, 2.5, 0.0, 0.0],      # C
        [6, 3.75, 2.2, 0.0],     # C (bent)
        [1, -1.0, 0.0, 0.0],     # H
        [1, 4.75, 2.2, 0.0],     # H
    ]

    # Ligand: same shape but rotated ~30 degrees and shifted
    angle = np.radians(30)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    shift = np.array([1.0, 0.5, 0.3])

    lig_atoms_bohr = []
    for atom in ref_atoms_bohr:
        x, y, z = atom[1], atom[2], atom[3]
        # Rotate around Z axis
        new_x = x * cos_a - y * sin_a + shift[0]
        new_y = x * sin_a + y * cos_a + shift[1]
        new_z = z + shift[2]
        lig_atoms_bohr.append([atom[0], new_x, new_y, new_z])

    # Create density grids
    grid_shape = (grid_size, grid_size, grid_size)

    ref_positions = [bohr_to_grid(np.array(a[1:4])) for a in ref_atoms_bohr]
    lig_positions = [bohr_to_grid(np.array(a[1:4])) for a in lig_atoms_bohr]

    ref_density = create_molecule_density(grid_shape, ref_positions, sigma=2.5)
    lig_density = create_molecule_density(grid_shape, lig_positions, sigma=2.5)

    # Normalize densities to reasonable values
    ref_density = ref_density / ref_density.max() * 0.1
    lig_density = lig_density / lig_density.max() * 0.1

    # Write cube files
    write_cube(
        "reference_test.cube",
        ref_atoms_bohr,
        ref_density,
        origin,
        voxel_size,
        "Reference test molecule",
        "Synthetic density for testing"
    )

    write_cube(
        "ligand_test.cube",
        lig_atoms_bohr,
        lig_density,
        origin,
        voxel_size,
        "Ligand test molecule",
        "Rotated and shifted version of reference"
    )

    print("Created test cube files:")
    print("  - reference_test.cube")
    print("  - ligand_test.cube")
    print()
    print("These are small synthetic files for testing.")
    print("The ligand is the reference rotated by 30 degrees and shifted.")
    print()
    print("Test with:")
    print("  density-align reference_test.cube ligand_test.cube -o test_output/")


if __name__ == "__main__":
    main()
