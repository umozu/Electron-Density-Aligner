# Density Aligner - Documentation

## What This Tool Does

Density Aligner takes two molecules represented as electron density grids (cube files) and finds the best way to overlay them. Unlike traditional shape-matching tools that approximate molecules as collections of spheres, this approach uses actual quantum mechanical electron densities computed by programs like ORCA, Gaussian, or Psi4.

The result is a more accurate alignment that captures not just the molecular shape but also the electronic structure - where electrons actually are in the molecule.

### Why Use Electron Density?

Most molecular alignment tools use simplified representations:
- **ROCS** uses Gaussian functions centered on atoms
- **Shape-it** uses atomic radii to define molecular volume
- **Pharmacophore methods** use feature points

These approximations work well for many applications, but they miss details. Electron density captures:
- Lone pairs and their directionality
- Pi electron clouds above/below aromatic rings
- Polarization effects from nearby groups
- The actual "shape" that other molecules interact with

## How It Works

### The Alignment Problem

Given two density grids A (reference) and B (ligand), we want to find a rotation R and translation t that maximizes the overlap between A and the transformed B:

```
maximize: Overlap(A, R·B + t)
```

The overlap is measured as a spatial overlap coefficient - basically "what fraction of occupied space is shared between the two densities":

```
Overlap = (A ∩ B) / (A ∪ B)
```

where A and B are the regions with non-zero density.

### The Algorithm

1. **Multi-start optimization**: Molecules can have similar shapes in different orientations. We try 4 starting positions:
   - Original orientation
   - 180° rotation around X axis
   - 180° rotation around Y axis
   - 180° rotation around both X and Y

2. **Random initialization**: For each starting position, we run multiple optimizations with different random seeds. This helps escape local minima.

3. **Gradient descent**: We use PyTorch's automatic differentiation to compute gradients of the overlap with respect to rotation and translation. The Adam optimizer updates the parameters to increase overlap.

4. **Early stopping**: Optimization stops when improvement drops below 0.01% over 10 iterations.

5. **Best result**: We keep the transformation that gives the highest overlap across all runs.

### Technical Details

- **Rotation representation**: We use quaternions (4 numbers) instead of Euler angles to avoid gimbal lock and ensure smooth optimization.

- **Density transformation**: The density grid is resampled using trilinear interpolation after applying the rotation and translation.

- **Soft thresholding**: For gradient computation, we use a sigmoid function to smoothly transition between "occupied" and "empty" space. This makes the overlap differentiable.

## Installation

```bash
# From PyPI (when published)
pip install density-aligner

# From source
git clone https://github.com/yourusername/density-aligner.git
cd density-aligner
pip install -e .
```

### Requirements

- Python 3.8 or newer
- NumPy
- PyTorch (CPU is fine, GPU speeds things up for large grids)
- SciPy

## Usage

### Command Line

Basic usage:
```bash
density-align reference.cube ligand.cube -o results/
```

All options:
```bash
density-align reference.cube ligand.cube \
    -o output_directory \    # where to save results
    -n 10 \                  # random starts per rotation (default: 10)
    -m 1000 \                # max iterations per run (default: 1000)
    -q                       # quiet mode, less output
```

Example with real files:
```bash
density-align pimozide.cube astemizole.cube -o alignment_results/
```

### Python API

```python
from density_aligner import read_cube, align_densities

# Load your cube files
ref = read_cube("reference.cube")
lig = read_cube("ligand.cube")

# Run alignment
result = align_densities(ref, lig, n_starts=10, verbose=True)

# Check results
print(f"Overlap: {result.true_binary_overlap:.3f}")
print(f"Best starting position: {result.pre_rotation}")
print(f"Rotation (quaternion): {result.quaternion}")
print(f"Translation (Bohr): {result.translation}")
```

For more control:
```python
from density_aligner import read_cube, write_cube, write_xyz
from density_aligner.spatial_aligner import (
    align_densities,
    transform_atoms,
    rotate_atoms_180
)

# Load data
ref = read_cube("reference.cube")
lig = read_cube("ligand.cube")

# Align with custom settings
result = align_densities(
    ref, lig,
    n_starts=5,           # fewer starts = faster but less thorough
    seeds=[42, 123, 456], # specific seeds for reproducibility
    max_iters=500,        # fewer iterations
    verbose=True
)

# Apply transformation to get aligned coordinates
if result.pre_rotation != 'original':
    atoms = rotate_atoms_180(lig.atoms, result.pre_rotation, lig.grid_center)
else:
    atoms = lig.atoms

aligned_atoms = transform_atoms(
    atoms,
    result.quaternion,
    result.translation,
    ref.grid_center
)

# Save aligned structure
write_xyz("aligned.xyz", aligned_atoms, title="Aligned molecule")
```

## Input Files

### Cube File Format

Cube files are a standard format from computational chemistry. They contain:
- Two title/comment lines
- Number of atoms and grid origin
- Grid dimensions and voxel vectors (3 lines)
- Atom coordinates (one line per atom)
- Density values (the rest of the file)

Example header:
```
Title line 1
Title line 2
   65  -32.000000  -32.000000  -32.000000    <- 65 atoms, origin at (-32,-32,-32)
  200    0.321600    0.000000    0.000000    <- 200 points in X, voxel size 0.32 Bohr
  200    0.000000    0.321600    0.000000    <- 200 points in Y
  200    0.000000    0.000000    0.321600    <- 200 points in Z
    6    0.000000   -5.123456    2.345678    0.987654  <- Carbon at (x,y,z)
    ... more atoms ...
    1.23456E-06  2.34567E-06  ...           <- density values
```

### Generating Cube Files

**With ORCA:**
```
! B3LYP def2-SVP
%plots
  dim1 100
  dim2 100
  dim3 100
  Format Cube
  ElDens("molecule.cube");
end
* xyz 0 1
C  0.0  0.0  0.0
... atom coordinates ...
*
```

**With Gaussian:**
```
#p B3LYP/6-31G* Cube=Full

Title

0 1
C  0.0  0.0  0.0
... atom coordinates ...
```

### Grid Size Considerations

- **Small grids (80³)**: Fast to compute and align, but may miss details
- **Medium grids (120³)**: Good balance for most applications
- **Large grids (200³)**: High accuracy but slow to align and large files (~100 MB)

Both molecules should use the **same grid dimensions** for best results. Different grid sizes will still work but alignment quality may suffer.

## Output Files

After running alignment, you get:

| File | Description |
|------|-------------|
| `*_reference.xyz` | Reference molecule coordinates (unchanged) |
| `*_aligned.xyz` | Ligand coordinates after alignment |
| `combined.xyz` | Both molecules in one file for visualization |
| `*_aligned.cube` | Full cube file with aligned density |
| `alignment_params.npz` | NumPy archive with transformation parameters |
| `alignment_report.txt` | Human-readable summary |

### Loading Saved Parameters

```python
import numpy as np

params = np.load("alignment_params.npz")
print(f"Quaternion: {params['quaternion']}")
print(f"Translation: {params['translation']}")
print(f"Pre-rotation: {params['pre_rotation']}")
print(f"Overlap: {params['binary_overlap']}")
```

## Visualizing Results

The `combined.xyz` file contains both molecules and can be opened in any molecular viewer:

- **PyMOL**: `load combined.xyz`
- **VMD**: `mol new combined.xyz`
- **Chimera**: File → Open
- **Avogadro**: File → Open

To visualize the density overlap, load both cube files:

**In PyMOL:**
```
load reference.cube
load aligned.cube
isosurface ref_surf, reference, 0.01
isosurface lig_surf, aligned, 0.01
set transparency, 0.5, lig_surf
```

## Interpreting Results

### Overlap Values

- **> 0.7**: Excellent overlap, molecules are very similar in shape
- **0.5 - 0.7**: Good overlap, significant shape similarity
- **0.3 - 0.5**: Moderate overlap, some shape features match
- **< 0.3**: Poor overlap, molecules have different shapes

### Common Issues

**Low overlap despite similar-looking molecules:**
- Try running with more random starts (`-n 20`)
- Check that both cube files have similar grid extents
- Make sure densities are computed at the same level of theory

**Alignment seems wrong visually:**
- The algorithm maximizes density overlap, not atom overlap
- Two molecules can have similar density shapes but different atom positions
- Check the density isosurfaces, not just atom positions

**Slow performance:**
- Larger grids take longer: O(n³) for n×n×n grid
- Use fewer random starts for quick preliminary alignments
- Consider using a GPU (PyTorch will use it automatically if available)

## Examples

### Example 1: Basic Alignment

```bash
# Align two drug molecules
density-align pimozide.cube astemizole.cube -o drug_alignment/

# Check the overlap
cat drug_alignment/alignment_report.txt
```

### Example 2: Batch Processing

```python
import os
from density_aligner import read_cube, align_densities
from density_aligner.cube_io import write_xyz

reference = read_cube("reference.cube")

for ligand_file in os.listdir("ligands/"):
    if ligand_file.endswith(".cube"):
        lig = read_cube(f"ligands/{ligand_file}")
        result = align_densities(reference, lig, n_starts=5, verbose=False)
        print(f"{ligand_file}: overlap = {result.true_binary_overlap:.3f}")
```

### Example 3: Quick vs Thorough Alignment

```python
from density_aligner import read_cube, align_densities

ref = read_cube("reference.cube")
lig = read_cube("ligand.cube")

# Quick alignment (for screening)
quick = align_densities(ref, lig, n_starts=2, max_iters=200)
print(f"Quick: {quick.true_binary_overlap:.3f}")

# Thorough alignment (for final results)
thorough = align_densities(ref, lig, n_starts=10, max_iters=1000)
print(f"Thorough: {thorough.true_binary_overlap:.3f}")
```

## API Reference

### Main Functions

**`read_cube(filename)`** - Read a cube file, returns `CubeData` object

**`write_cube(filename, cube_data)`** - Write a cube file

**`write_xyz(filename, atoms, title="")`** - Write atoms to XYZ format

**`align_densities(ref, lig, n_starts=10, seeds=None, max_iters=1000, verbose=True)`** - Main alignment function, returns `AlignmentResult`

### Data Classes

**`CubeData`** - Container for cube file data
- `.atoms` - List of `Atom` objects
- `.density` - 3D numpy array
- `.origin` - Grid origin (Bohr)
- `.voxel_vectors` - Tuple of 3 vectors defining the grid
- `.grid_shape` - Tuple (nx, ny, nz)
- `.grid_center` - Center of the grid

**`Atom`** - Single atom
- `.atomic_num` - Atomic number
- `.coords` - Coordinates in Bohr
- `.coords_angstrom` - Coordinates in Angstrom
- `.symbol` - Element symbol (C, N, O, etc.)

**`AlignmentResult`** - Alignment output
- `.quaternion` - Rotation as [w, x, y, z]
- `.translation` - Translation vector (Bohr)
- `.spatial_overlap` - Soft overlap (used during optimization)
- `.true_binary_overlap` - Actual binary overlap (final metric)
- `.pre_rotation` - Which 180° rotation was best ('original', '180_x', '180_y', '180_xy')
- `.seed` - Random seed of best run
- `.iterations` - Number of iterations

## Troubleshooting

### "CUDA out of memory"
Large grids (200³+) may exceed GPU memory. Either:
- Use CPU: `CUDA_VISIBLE_DEVICES="" density-align ...`
- Use smaller grid resolution in your quantum chemistry calculation

### "Grid shapes differ" warning
The code works with different grid sizes but results are better when they match. Regenerate cube files with the same grid parameters.

### Very slow alignment
- Check if PyTorch is using GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce grid size in quantum chemistry calculation
- Use fewer random starts for preliminary screening

## License

MIT License - free for academic and commercial use.

## Citation

If you use this tool in published work, please cite:

```
Density Aligner: Molecular alignment using electron density overlap
https://github.com/yourusername/density-aligner
```
