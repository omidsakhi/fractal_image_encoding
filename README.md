# fractal_image_encoding

A from-scratch, NumPy-only implementation of **fractal image compression**: encode a grayscale image as a set of contractive affine maps, then reconstruct it by iterating those maps from a random seed until they converge to the image's attractor.

A study / exploratory project — a compact, dependency-light take on Jacquin–Fisher-style fractal coding with an adaptive quadtree partition.

## Background

Fractal compression models an image as the fixed point of a **partitioned iterated function system** (PIFS). Each range block in the image is approximated by an affine transform of some (larger) domain block taken from the same image:

```
R(x, y) ≈ s · D(x, y) + b
```

where `s` is a contractive scale factor and `b` is a brightness offset. The collection of such maps forms a contractive operator on image space, so by Banach's fixed-point theorem, iterating the maps from any starting image converges to a unique attractor that approximates the original.

## How it works

### Encoder (`encoder.py`)

- Builds the **domain pool** from the 8 isometries of the D4 dihedral group (4 rotations + horizontal flip + 4 rotations of the flip) applied to the source image.
- Extracts candidate domain blocks at multiple sizes (up to 64×64) and downsample factors (2, 4, 8, 16), so each domain can be matched against range blocks that are 1/2, 1/4, 1/8, or 1/16 its linear size.
- Starts with a uniform grid of 32×32 range blocks. For each one, finds the best-fitting domain by **closed-form least squares** for `s` and `b`, with:
  - contractivity enforced: `s ∈ [0, 0.95]`, quantized to 8 levels
  - brightness: `b ∈ [0, 1]`, quantized to 100 levels
- If the best match exceeds the error threshold, the range block is **quadtree-split** into four sub-blocks and each is re-matched independently. Splitting stops at a 4×4 floor.
- Writes the resulting transform table to CSV — one row per range block, storing domain coords/size, downsample scale, transform parameters, and fit error.

### Decoder (`decoder.py`)

- Starts from a random image of the target size.
- Each iteration: rebuilds the 8-fold D4 domain pool from the current estimate, applies every transform in the CSV to fill the corresponding range block, then replaces the estimate with the result.
- After a handful of iterations the estimate converges to the attractor of the map set — a reconstruction of the encoded image.

## Layout

```
fractal_image_encoding/
├── main.py             CLI: --encode / --decode
├── encoder.py          Encoder + Transformation record + closed-form affine fit
├── decoder.py          Decoder (Banach fixed-point iteration from random seed)
├── integral.py         Summed-area table helper
├── utils.py            Domain pyramid helpers (D4 isometries × downsample levels)
└── requirments.txt     Declared deps (see Dependencies below)
```

## Usage

```bash
# Encode a grayscale image to a CSV of transforms
python main.py --encode path/to/image.png --output transforms.csv \
    --error_threshold 0.01 --verbose

# Decode the CSV back to an image via fixed-point iteration
python main.py --decode transforms.csv --output reconstructed.png \
    --iterations 6
```

The encoder converts RGB inputs to grayscale via `Image.convert('L')`. The decoder currently assumes a 256×256 output.

## Dependencies

```
numpy
Pillow
tqdm
pandas
```

(`requirments.txt` lists only `numpy` and `Pillow`; `tqdm` and `pandas` are also imported by the encoder/decoder.)

## Notes

- Uses the standard Jacquin / Fisher framing with the D4 symmetry group on the domain pool — 8 isometries per size class.
- Both `s` and `b` are quantized; this is the typical route to turning a continuous affine map into a cheap, compact record.
- Pure NumPy — no OpenCV, no torch, no GPU. The search is O(|range blocks| · |domain blocks|) within each size class, so this is a readable study implementation, not a production codec. Expect a few minutes to encode a 256×256 image.
- The `IntegralImage` helper (in `integral.py`, and duplicated inside `encoder.py`) is wired in as a potential optimization for fast block-sum queries but isn't on the active match path.
