import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd


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
    def __init__(self, image_path, max_domain_size=64, max_range_size=32, min_range_size=4, error_threshold=0.01, verbose=False):
        self.max_domain_size = max_domain_size
        self.max_range_size = max_range_size
        self.min_range_size = min_range_size
        self.error_threshold = error_threshold        
        self.verbose = verbose
        
        self.image = np.array(Image.open(image_path).convert('L')) / 255.0

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
    
    def compute(self, range_block, domain_block):        
        n = range_block.size
        sum_r = np.sum(range_block)
        sum_d = np.sum(domain_block)
        sum_d2 = np.sum(domain_block**2)
        sum_dr = np.sum(domain_block * range_block)

        scale = (n * sum_dr - sum_d * sum_r) / (n * sum_d2 - sum_d**2)       
        scale = np.clip(scale, 0, 0.95)
        scale = np.round(scale * 7 / 0.95) * 0.95 / 7

        bias = (sum_r - scale * sum_d) / n
        bias = np.clip(bias, 0, 1)
        bias = np.round(bias * 100) / 100

        error = np.mean((range_block - (domain_block * scale + bias))**2)        

        return scale, bias, error

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
                            domain_block = self.domain_images[rot][y:y+size, x:x+size]
                            if scale > 1:
                                domain_block = domain_block.reshape(size//scale, scale, size//scale, scale).mean(axis=(1,3))
                            self.info_domain_blocks.append((rot, x, y, size, scale))
            size = size // 2
        
        assert len(self.info_domain_blocks) > 0

        if self.verbose:
            print(f"Extracted {len(self.info_domain_blocks)} domain blocks")

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
        transforms = []
        coverage = np.zeros_like(self.image)
        total_pixels = np.prod(self.image.shape)
        
        with tqdm(total=100, desc="Encoding Progress", unit="%") as pbar:
            while self.info_range_blocks:                   
                current_range_info = self.info_range_blocks.pop()            
                rx, ry, rsize = current_range_info            
                range_block = self.image[ry:ry+rsize, rx:rx+rsize]            
                best_transform = None
                best_error = float('inf')                 
                filtered_domain_blocks = list(filter(lambda x: x[3] // x[4] == rsize, self.info_domain_blocks))                
                assert len(filtered_domain_blocks) > 0
                for domain_block_info in filtered_domain_blocks:
                    di, dx, dy, dsize, dscale = domain_block_info                
                    domain_block = self.domain_images[di][dy:dy+dsize, dx:dx+dsize]
                    if dscale > 1:
                        domain_block = domain_block.reshape(dsize//dscale, dscale, dsize//dscale, dscale).mean(axis=(1,3))                
                    scale, bias, error = self.compute(range_block, domain_block)
                    if error < best_error:
                        best_error = error
                        transform = Transformation()
                        transform.domain_source_index = di
                        transform.domain_x = dx
                        transform.domain_y = dy
                        transform.domain_size = dsize
                        transform.domain_scale = dscale
                        transform.range_x = rx
                        transform.range_y = ry
                        transform.range_size = rsize
                        transform.scale = scale
                        transform.bias = bias
                        transform.error = error
                        best_transform = transform
                    
                    if best_error <= self.error_threshold:
                        break

                if best_error <= self.error_threshold or rsize <= self.min_range_size:
                    transforms.append(best_transform)
                    coverage[ry:ry+rsize, rx:rx+rsize] = 1
                    current_coverage = np.sum(coverage) / total_pixels * 100
                    pbar.n = int(current_coverage)
                    pbar.refresh()
                else:
                    half_size = rsize // 2
                    self.info_range_blocks.extend([
                        (rx, ry, half_size),
                        (rx + half_size, ry, half_size),
                        (rx, ry + half_size, half_size),
                        (rx + half_size, ry + half_size, half_size)
                    ])

        if self.verbose:
            print(f"Found {len(transforms)} transforms")

        return transforms

    def save_transforms_to_csv(self, transforms, filename='transforms.csv'):
        
        # Create a list of dictionaries, each representing a transform
        transform_data = []
        for t in transforms:
            transform_data.append({
                'domain_source_index': t.domain_source_index,
                'domain_x': t.domain_x,
                'domain_y': t.domain_y,
                'domain_size': t.domain_size,
                'domain_scale': t.domain_scale,
                'range_x': t.range_x,
                'range_y': t.range_y,
                'range_size': t.range_size,
                'scale': t.scale,
                'bias': t.bias,
                'error': t.error
            })
        
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(transform_data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(filename, index=False)
        
        if self.verbose:
            print(f"Transforms saved to {filename}")

