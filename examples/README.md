# Example Files

## Quick Start

Generate small test cube files:

```bash
cd examples
python generate_test_cubes.py
```

This creates `reference_test.cube` and `ligand_test.cube` - small synthetic densities for testing.

Then run alignment:

```bash
density-align reference_test.cube ligand_test.cube -o test_output/
```

## Using Real Cube Files

For real applications, you need electron density cube files from quantum chemistry calculations. These files can be large (50-200 MB for high-resolution grids).

### Generating with ORCA

Example ORCA input to generate a cube file:

```
! B3LYP def2-SVP TightSCF

%plots
  dim1 120
  dim2 120
  dim3 120
  Format Cube
  ElDens("molecule_density.cube");
end

* xyzfile 0 1 molecule.xyz
```

Run with:
```bash
orca molecule.inp > molecule.out
```

### Generating with Gaussian

Example Gaussian input:

```
%chk=molecule.chk
#p B3LYP/6-31G* Cube=Full Density=Current

Title

0 1
C  0.000000  0.000000  0.000000
... more atoms ...

molecule.cube
```

### Tips for Cube Files

1. **Use the same grid for both molecules** - both should have the same number of points and voxel size for best results.

2. **Grid size tradeoffs:**
   - 80×80×80: Fast, ~10 MB files
   - 120×120×120: Good balance, ~30 MB files
   - 200×200×200: High quality, ~100 MB files

3. **Make sure the grid is large enough** to contain the entire molecule with some padding.

## Example Results

With the test files, you should see output like:

```
======================================================================
DENSITY ALIGNER
======================================================================

Reference: reference_test.cube
Ligand:    ligand_test.cube
Output:    test_output

Reference: 5 atoms, grid (50, 50, 50)
Ligand:    5 atoms, grid (50, 50, 50)

Starting alignment (40 optimizations)...
----------------------------------------------------------------------
...
----------------------------------------------------------------------

Alignment complete in X seconds

Best result:
  Pre-rotation:     original
  Binary overlap:   0.85+
```

The high overlap confirms the code correctly identified that the ligand is just a rotated/shifted version of the reference.
