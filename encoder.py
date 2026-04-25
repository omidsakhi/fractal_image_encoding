import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from tqdm import tqdm


class Transformation:
    domain_source_index = 0
    domain_x = -1
    domain_y = -1
    domain_size = -1
    domain_scale = 1
    range_x = -1
    range_y = -1
    range_size = -1
    scale = 1
    bias = 0
    error = 0

class IntegralImage:
    def __init__(self, image):
        self.integral_image = np.zeros_like(image)
        self.compute(image)

    def compute(self, image):
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                self.integral_image[y, x] = image[y, x] + self.integral_image[y-1, x] + self.integral_image[y, x-1] - self.integral_image[y-1, x-1]

    def sum_region(self, x1, y1, x2, y2):
        return self.integral_image[y2, x2] - self.integral_image[y1, x2] - self.integral_image[y2, x1] + self.integral_image[y1, x1]

class Encoder:
    def __init__(self, image_path, max_domain_size=64, max_range_size=32, min_range_size=4, error_threshold=0.01, batch_size=256, num_workers=1, verbose=False):
        """
        num_workers: number of threads used to process range-block chunks in
            parallel within a size group. Defaults to 1 (serial). When raising
            this, prefer to also pin BLAS to a single thread per chunk
            (e.g. set OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1 /
            OPENBLAS_NUM_THREADS=1 in the environment) to avoid oversubscribing
            cores, since np.tensordot is often already multi-threaded.
        """
        self.max_domain_size = max_domain_size
        self.max_range_size = max_range_size
        self.min_range_size = min_range_size
        self.error_threshold = error_threshold
        self.batch_size = batch_size
        self.num_workers = max(1, int(num_workers))
        self.verbose = verbose
        
        self.image = np.array(Image.open(image_path).convert('L')) / 255.0
        h, w = self.image.shape
        if h % self.max_range_size != 0 or w % self.max_range_size != 0:
            raise ValueError(
                f"Image size {w}x{h} is not a multiple of max_range_size={self.max_range_size}. "
                f"The initial range grid would leave edge pixels uncovered; use a size like "
                f"{w - (w % self.max_range_size)}x{h - (h % self.max_range_size)} (crop) or "
                f"pad to the next multiple of {self.max_range_size}."
            )

        self.domain_images = [
            self.image,
            np.rot90(self.image),
            np.rot90(self.image, 2),
            np.rot90(self.image, 3),
            np.flipud(self.image),
            np.rot90(np.flipud(self.image)),
            np.rot90(np.flipud(self.image), 2),
            np.rot90(np.flipud(self.image), 3),                        
        ]

        self.info_domain_blocks = [] # List of (rot, x, y, size, scale) where scale 1 is the original size and scale 2 is half the size and rot is the rotated image that domain is extracted from between 1 to 10        
        self.extract_domain_blocks()

        self.info_range_blocks = []
        self.extract_range_blocks()

        # Cached per-output-size arrays of downsampled domain blocks plus their precomputed sums.
        # Built lazily on first encode() call so construction stays cheap.
        self._domain_cache = None

    def extract_range_blocks(self):
        
        height, width = self.image.shape

        range_size = self.max_range_size
        for y in range(0, height-range_size+1, range_size):
            for x in range(0, width-range_size+1, range_size):
                self.info_range_blocks.append((x, y, range_size))                

        if self.verbose:
            print(f"Extracted {len(self.info_range_blocks)} range blocks")

    def extract_domain_blocks(self):
        
        height, width = self.image.shape

        size = self.max_domain_size
        while size >= self.min_range_size * 2:
            for scale in [2, 4, 8, 16]:
                if scale > size // self.min_range_size:
                    break
                step = size // scale
                for y in range(0, height - size + 1, step):
                    for x in range(0, width - size + 1, step):
                        for rot in range(len(self.domain_images)):
                            self.info_domain_blocks.append((rot, x, y, size, scale))
            size = size // 2
        
        assert len(self.info_domain_blocks) > 0

        if self.verbose:
            print(f"Extracted {len(self.info_domain_blocks)} domain blocks")

    def _build_domain_cache(self):
        """Group domain blocks by their downsampled output size and precompute
        the actual block arrays plus per-block sums. This is what allows the
        encode loop to vectorize across all domain candidates for a range size.
        """
        cache = {}
        by_size = {}
        for info in self.info_domain_blocks:
            di, dx, dy, dsize, dscale = info
            out_size = dsize // dscale
            by_size.setdefault(out_size, []).append(info)

        for out_size, infos in by_size.items():
            blocks = np.empty((len(infos), out_size, out_size), dtype=self.image.dtype)
            for i, (di, dx, dy, dsize, dscale) in enumerate(infos):
                d = self.domain_images[di][dy:dy+dsize, dx:dx+dsize]
                if dscale > 1:
                    d = d.reshape(out_size, dscale, out_size, dscale).mean(axis=(1, 3))
                blocks[i] = d
            sum_d = blocks.sum(axis=(1, 2))
            sum_d2 = (blocks * blocks).sum(axis=(1, 2))
            cache[out_size] = {
                'blocks': blocks,
                'info': infos,
                'sum_d': sum_d,
                'sum_d2': sum_d2,
            }

        self._domain_cache = cache

        if self.verbose:
            for out_size, entry in cache.items():
                print(f"  cached {len(entry['info'])} domain blocks at output size {out_size}")

    def save_domain_sources(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, rotation in enumerate(self.domain_images):
            img = Image.fromarray((rotation * 255).astype(np.uint8))
            filename = f"rotation_{i}.png"
            img.save(os.path.join(output_dir, filename))
        
        print(f"Saved {len(self.domain_images)} rotations to {output_dir}")

    def print_transform(self, transform):
        print("--------------------------------")
        print("domain_source_index: ", transform.domain_source_index)
        print("domain_x: ", transform.domain_x)
        print("domain_y: ", transform.domain_y)
        print("domain_size: ", transform.domain_size)
        print("domain_scale: ", transform.domain_scale)
        print("range_x: ", transform.range_x)
        print("range_y: ", transform.range_y)
        print("range_size: ", transform.range_size)
        print("scale: ", transform.scale)
        print("bias: ", transform.bias)
        print("error: ", transform.error)

    def encode(self):
        if self._domain_cache is None:
            self._build_domain_cache()

        transforms = []
        coverage = np.zeros_like(self.image)
        total_pixels = np.prod(self.image.shape)

        # Work queue of range blocks waiting to be matched. Adaptive subdivision
        # appends smaller children here.
        pending = list(self.info_range_blocks)
        self.info_range_blocks = []

        with tqdm(total=100, desc="Encoding Progress", unit="%") as pbar:
            while pending:
                # Group pending range blocks by size so we can share the cached
                # domain array (and one big tensordot) across each batch.
                by_size = {}
                for info in pending:
                    by_size.setdefault(info[2], []).append(info)
                pending = []

                for rsize, infos in by_size.items():
                    domain = self._domain_cache.get(rsize)
                    assert domain is not None, f"no domain blocks cached for range size {rsize}"

                    new_pending, new_transforms = self._encode_size_group(
                        rsize, infos, domain, coverage,
                    )
                    transforms.extend(new_transforms)
                    pending.extend(new_pending)

                    current_coverage = np.sum(coverage) / total_pixels * 100
                    pbar.n = int(current_coverage)
                    pbar.refresh()

        if self.verbose:
            print(f"Found {len(transforms)} transforms")

        return transforms

    def _encode_size_group(self, rsize, range_infos, domain, coverage):
        """Match every range block in `range_infos` (all of size `rsize`) against
        every cached domain block of the same output size. Returns the list of
        accepted Transformations plus any range blocks that need to be subdivided.

        Range blocks are processed in chunks of `self.batch_size` to bound the
        (K, M) intermediate. When `self.num_workers > 1`, chunks are dispatched
        to a thread pool; the heavy lifting (`np.tensordot` plus elementwise
        ops) releases the GIL so threads make progress in parallel.
        """
        n = rsize * rsize
        denom = n * domain['sum_d2'] - domain['sum_d'] ** 2  # (M,)
        denom_safe = np.where(denom == 0, 1.0, denom)
        ctx = {
            'rsize': rsize,
            'n': n,
            'D': domain['blocks'],
            'sum_d': domain['sum_d'],
            'sum_d2': domain['sum_d2'],
            'domain_info': domain['info'],
            'denom': denom,
            'denom_safe': denom_safe,
        }

        chunks = [
            range_infos[i:i + self.batch_size]
            for i in range(0, len(range_infos), self.batch_size)
        ]

        if self.num_workers > 1 and len(chunks) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(ex.map(lambda c: self._process_range_chunk(c, ctx), chunks))
        else:
            results = [self._process_range_chunk(c, ctx) for c in chunks]

        # Merge chunk results back on the main thread so coverage and the
        # transform list don't need locking inside the workers.
        transforms = []
        next_pending = []
        for chunk_transforms, chunk_pending, chunk_accepted in results:
            transforms.extend(chunk_transforms)
            next_pending.extend(chunk_pending)
            for rx, ry in chunk_accepted:
                coverage[ry:ry + rsize, rx:rx + rsize] = 1

        return next_pending, transforms

    def _process_range_chunk(self, chunk, ctx):
        """Pure worker: matches one chunk of range blocks (all the same size)
        against the cached domain pool. Returns (transforms, subdivided_pending,
        accepted_coords) without touching any Encoder state, so it is safe to
        run from a worker thread.
        """
        rsize = ctx['rsize']
        n = ctx['n']
        D = ctx['D']
        sum_d = ctx['sum_d']
        sum_d2 = ctx['sum_d2']
        domain_info = ctx['domain_info']
        denom = ctx['denom']
        denom_safe = ctx['denom_safe']

        K = len(chunk)
        R = np.empty((K, rsize, rsize), dtype=self.image.dtype)
        for i, (rx, ry, _) in enumerate(chunk):
            R[i] = self.image[ry:ry + rsize, rx:rx + rsize]

        sum_r = R.sum(axis=(1, 2))                          # (K,)
        sum_r2 = (R * R).sum(axis=(1, 2))                   # (K,)
        sum_dr = np.tensordot(R, D, axes=([1, 2], [1, 2]))  # (K, M)

        # Same regression as the original compute(), just broadcast over (K, M).
        num = n * sum_dr - sum_d[None, :] * sum_r[:, None]
        scale = np.where(denom[None, :] == 0, 0.0, num / denom_safe[None, :])
        scale = np.clip(scale, 0.0, 0.95)
        bias = np.where(
            denom[None, :] == 0,
            sum_r[:, None] / n,
            (sum_r[:, None] - scale * sum_d[None, :]) / n,
        )

        # Quantization (matches original compute() exactly).
        scale = np.round(scale * 7 / 0.95) * 0.95 / 7
        bias = np.clip(bias, 0.0, 1.0)
        bias = np.round(bias * 100) / 100

        # Closed-form MSE from the precomputed sums.
        error = (
            sum_r2[:, None]
            - 2.0 * scale * sum_dr
            - 2.0 * bias * sum_r[:, None]
            + scale ** 2 * sum_d2[None, :]
            + 2.0 * scale * bias * sum_d[None, :]
            + n * bias ** 2
        ) / n

        best_m = np.argmin(error, axis=1)
        row_idx = np.arange(K)
        best_error = error[row_idx, best_m]
        best_scale = scale[row_idx, best_m]
        best_bias = bias[row_idx, best_m]

        transforms = []
        next_pending = []
        accepted = []
        for k, (rx, ry, _) in enumerate(chunk):
            m = int(best_m[k])
            err = float(best_error[k])

            if err <= self.error_threshold or rsize <= self.min_range_size:
                di, dx, dy, dsize, dscale = domain_info[m]
                t = Transformation()
                t.domain_source_index = di
                t.domain_x = dx
                t.domain_y = dy
                t.domain_size = dsize
                t.domain_scale = dscale
                t.range_x = rx
                t.range_y = ry
                t.range_size = rsize
                t.scale = float(best_scale[k])
                t.bias = float(best_bias[k])
                t.error = err
                transforms.append(t)
                accepted.append((rx, ry))
            else:
                half = rsize // 2
                next_pending.extend([
                    (rx, ry, half),
                    (rx + half, ry, half),
                    (rx, ry + half, half),
                    (rx + half, ry + half, half),
                ])

        return transforms, next_pending, accepted

    def save_transforms(self, transforms, filename='transforms.json'):
        image_height, image_width = self.image.shape
        records = [
            {
                'domain_source_index': int(t.domain_source_index),
                'domain_x': int(t.domain_x),
                'domain_y': int(t.domain_y),
                'domain_size': int(t.domain_size),
                'domain_scale': int(t.domain_scale),
                'range_x': int(t.range_x),
                'range_y': int(t.range_y),
                'range_size': int(t.range_size),
                'scale': float(t.scale),
                'bias': float(t.bias),
                'error': float(t.error),
            }
            for t in transforms
        ]
        doc = {
            'format': 'fractal-pifs',
            'version': 1,
            'image_height': int(image_height),
            'image_width': int(image_width),
            'transforms': records,
        }
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                json.dump(doc, f)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2)

        if self.verbose:
            print(f"Transforms saved to {filename}")

