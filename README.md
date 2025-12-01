# Density Aligner

Align molecules using quantum mechanical electron density overlap.

## What is this?

This tool aligns two molecules by finding the rotation and translation that maximizes the overlap of their electron densities. Unlike shape-matching methods that use spheres or Gaussians to approximate molecular shape, this uses actual electron density from quantum chemistry calculations.

The electron density captures the real "shape" of a molecule - including lone pairs, pi clouds, and polarization effects that simplified representations miss.

## Installation

```bash
pip install density-aligner
```

Or from source:
```bash
git clone https://github.com/yourusername/density-aligner.git
cd density-aligner
pip install -e .
```

## Quick Start

```bash
# Align ligand to reference
density-align reference.cube ligand.cube -o results/
```

This produces:
- `combined.xyz` - Both molecules for visualization
- `*_aligned.cube` - Aligned density
- `alignment_report.txt` - Summary with overlap score

## Example

```bash
# Generate test files
cd examples
python generate_test_cubes.py

# Run alignment
density-align reference_test.cube ligand_test.cube -o output/

# Check results
cat output/alignment_report.txt
```

## How It Works

1. Tries 4 starting orientations (to handle molecular symmetry)
2. For each, runs gradient descent to maximize density overlap
3. Uses multiple random seeds to avoid local minima
4. Returns the best alignment found

The overlap metric is:

```
Overlap = (Volume where both densities are non-zero) / (Volume where either is non-zero)
```

Values range from 0 (no overlap) to 1 (identical).

## Input Files

You need Gaussian cube files containing electron density. Generate these with:

**ORCA:**
```
! B3LYP def2-SVP
%plots
  dim1 120
  dim2 120
  dim3 120
  Format Cube
  ElDens("density.cube");
end
* xyzfile 0 1 molecule.xyz
```

**Gaussian:**
```
#p B3LYP/6-31G* Cube=Full Density=Current
```

Both molecules should use the same grid dimensions for best results.

## Python API

```python
from density_aligner import read_cube, align_densities

ref = read_cube("reference.cube")
lig = read_cube("ligand.cube")

result = align_densities(ref, lig)

print(f"Overlap: {result.true_binary_overlap:.3f}")
```

## Command Line Options

```
density-align reference.cube ligand.cube [options]

Options:
  -o, --output DIR     Output directory (default: alignment_output)
  -n, --n-starts N     Random starts per rotation (default: 10)
  -m, --max-iters N    Max iterations per run (default: 1000)
  -q, --quiet          Less output
```

## Typical Results

| Overlap | Interpretation |
|---------|---------------|
| > 0.7   | Very similar shapes |
| 0.5-0.7 | Good similarity |
| 0.3-0.5 | Some features match |
| < 0.3   | Different shapes |

## Requirements

- Python 3.8+
- NumPy
- PyTorch
- SciPy

## Documentation

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for detailed documentation including:
- Algorithm explanation
- API reference
- Troubleshooting
- Examples

## License

MIT
