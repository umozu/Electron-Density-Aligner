"""
Cube file I/O.

Functions for reading and writing Gaussian cube files. These files contain
both the molecular structure (atom positions) and a 3D grid of electron
density values. The format is pretty simple but has some quirks - coordinates
are in Bohr (atomic units), and the grid can be quite large for high resolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Atomic number to element symbol mapping
ATOMIC_SYMBOLS = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
    9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
    16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I'
}

# Element symbol to atomic number mapping
ATOMIC_NUMBERS = {v: k for k, v in ATOMIC_SYMBOLS.items()}


@dataclass
class Atom:
    """Represents an atom with its properties."""
    atomic_num: int
    charge: float
    coords: np.ndarray  # Coordinates in Bohr

    @property
    def symbol(self) -> str:
        return ATOMIC_SYMBOLS.get(self.atomic_num, 'X')

    @property
    def coords_angstrom(self) -> np.ndarray:
        return self.coords * 0.529177


@dataclass
class CubeData:
    """Container for cube file data."""
    title1: str
    title2: str
    atoms: List[Atom]
    origin: np.ndarray
    voxel_vectors: Tuple[np.ndarray, np.ndarray, np.ndarray]
    grid_shape: Tuple[int, int, int]
    density: np.ndarray

    @property
    def grid_center(self) -> np.ndarray:
        """Calculate the center of the grid."""
        nx, ny, nz = self.grid_shape
        vx, vy, vz = self.voxel_vectors
        grid_max = self.origin + (nx-1)*vx + (ny-1)*vy + (nz-1)*vz
        return (self.origin + grid_max) / 2

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    def get_atom_coords(self) -> np.ndarray:
        """Get all atom coordinates as an array."""
        return np.array([atom.coords for atom in self.atoms])

    def get_atom_coords_angstrom(self) -> np.ndarray:
        """Get all atom coordinates in Angstrom."""
        return self.get_atom_coords() * 0.529177


def read_cube(filename: str) -> CubeData:
    """
    Read a Gaussian cube file.

    Parameters
    ----------
    filename : str
        Path to the cube file.

    Returns
    -------
    CubeData
        Container with all cube file data.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    title1 = lines[0].strip()
    title2 = lines[1].strip()

    # Line 3: number of atoms and origin
    parts = lines[2].split()
    n_atoms = int(parts[0])
    origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    # Lines 4-6: grid dimensions and voxel vectors
    nx_parts = lines[3].split()
    nx = int(nx_parts[0])
    voxel_x = np.array([float(nx_parts[1]), float(nx_parts[2]), float(nx_parts[3])])

    ny_parts = lines[4].split()
    ny = int(ny_parts[0])
    voxel_y = np.array([float(ny_parts[1]), float(ny_parts[2]), float(ny_parts[3])])

    nz_parts = lines[5].split()
    nz = int(nz_parts[0])
    voxel_z = np.array([float(nz_parts[1]), float(nz_parts[2]), float(nz_parts[3])])

    # Parse atoms
    atoms = []
    for i in range(n_atoms):
        parts = lines[6 + i].split()
        atomic_num = int(parts[0])
        charge = float(parts[1])
        coords = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        atoms.append(Atom(atomic_num, charge, coords))

    # Parse density data
    density_values = []
    for line in lines[6 + n_atoms:]:
        density_values.extend([float(x) for x in line.split()])

    density = np.array(density_values[:nx * ny * nz]).reshape((nx, ny, nz))

    return CubeData(
        title1=title1,
        title2=title2,
        atoms=atoms,
        origin=origin,
        voxel_vectors=(voxel_x, voxel_y, voxel_z),
        grid_shape=(nx, ny, nz),
        density=density
    )


def write_cube(filename: str, cube_data: CubeData, title1: Optional[str] = None,
               title2: Optional[str] = None) -> None:
    """
    Write a Gaussian cube file.

    Parameters
    ----------
    filename : str
        Output filename.
    cube_data : CubeData
        Cube data to write.
    title1 : str, optional
        First title line (uses cube_data.title1 if not provided).
    title2 : str, optional
        Second title line (uses cube_data.title2 if not provided).
    """
    t1 = title1 if title1 is not None else cube_data.title1
    t2 = title2 if title2 is not None else cube_data.title2

    nx, ny, nz = cube_data.grid_shape
    vx, vy, vz = cube_data.voxel_vectors

    with open(filename, 'w') as f:
        # Title lines
        f.write(f"{t1}\n")
        f.write(f"{t2}\n")

        # Number of atoms and origin
        f.write(f"{cube_data.n_atoms:5d} {cube_data.origin[0]:12.6f} "
                f"{cube_data.origin[1]:12.6f} {cube_data.origin[2]:12.6f}\n")

        # Grid dimensions and voxel vectors
        f.write(f"{nx:5d} {vx[0]:12.6f} {vx[1]:12.6f} {vx[2]:12.6f}\n")
        f.write(f"{ny:5d} {vy[0]:12.6f} {vy[1]:12.6f} {vy[2]:12.6f}\n")
        f.write(f"{nz:5d} {vz[0]:12.6f} {vz[1]:12.6f} {vz[2]:12.6f}\n")

        # Atoms
        for atom in cube_data.atoms:
            f.write(f"{atom.atomic_num:5d} {atom.charge:12.6f} "
                    f"{atom.coords[0]:12.6f} {atom.coords[1]:12.6f} "
                    f"{atom.coords[2]:12.6f}\n")

        # Density data (6 values per line, scientific notation)
        values = cube_data.density.flatten()
        for i in range(0, len(values), 6):
            chunk = values[i:i+6]
            line = " ".join(f"{v:12.5E}" for v in chunk)
            f.write(line + "\n")


def write_xyz(filename: str, atoms: List[Atom], title: str = "molecule") -> None:
    """
    Write atoms to XYZ file format.

    Parameters
    ----------
    filename : str
        Output filename.
    atoms : List[Atom]
        List of atoms to write.
    title : str
        Title/comment line.
    """
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{title}\n")
        for atom in atoms:
            coords = atom.coords_angstrom
            f.write(f"{atom.symbol:2s} {coords[0]:12.6f} "
                    f"{coords[1]:12.6f} {coords[2]:12.6f}\n")


def write_combined_xyz(filename: str, ref_atoms: List[Atom],
                       lig_atoms: List[Atom], title: str = "combined") -> None:
    """
    Write two molecules to a single XYZ file for visualization.

    Parameters
    ----------
    filename : str
        Output filename.
    ref_atoms : List[Atom]
        Reference molecule atoms.
    lig_atoms : List[Atom]
        Ligand molecule atoms.
    title : str
        Title/comment line.
    """
    all_atoms = ref_atoms + lig_atoms
    write_xyz(filename, all_atoms, title)
