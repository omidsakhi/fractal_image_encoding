import numpy as np
from PIL import Image
import pandas as pd

class Decoder:
    def __init__(self, transforms_path, image_size, iterations=20):
        self.transforms_path = transforms_path
        self.image_size = image_size
        self.transforms = pd.read_csv(transforms_path)
        self.iterations = iterations        
    
    def decode(self):
        decoded = np.random.rand(*self.image_size)
        self.domain_images = [
            decoded,
            np.rot90(decoded),
            np.rot90(decoded, 2),
            np.rot90(decoded, 3),
            np.flipud(decoded),
            np.rot90(np.flipud(decoded)),
            np.rot90(np.flipud(decoded), 2),
            np.rot90(np.flipud(decoded), 3),                        
        ]
        for _ in range(self.iterations):
            new_decoded = np.zeros_like(decoded)
            for index, transform in self.transforms.iterrows():
                domain_source_index = int(transform.domain_source_index)
                domain_x = int(transform.domain_x)
                domain_y = int(transform.domain_y)
                domain_size = int(transform.domain_size)
                domain_scale = int(transform.domain_scale)
                range_x = int(transform.range_x)
                range_y = int(transform.range_y)
                range_size = int(transform.range_size)
                scale = float(transform.scale)
                bias = float(transform.bias)

                #print(domain_source_index, domain_x, domain_y, domain_size, domain_scale, range_x, range_y, range_size, scale, bias)
                domain_block = self.domain_images[domain_source_index][domain_y:domain_y+domain_size, domain_x:domain_x+domain_size]
                if domain_scale > 1:
                    domain_block = domain_block.reshape(domain_size//domain_scale, domain_scale, domain_size//domain_scale, domain_scale).mean(axis=(1,3))
                range_block = domain_block * scale + bias
                new_decoded[range_y:range_y+range_size, range_x:range_x+range_size] = range_block
            decoded = new_decoded
            self.domain_images = [
                decoded,
                np.rot90(decoded),
                np.rot90(decoded, 2),
                np.rot90(decoded, 3),
                np.flipud(decoded),
                np.rot90(np.flipud(decoded)),
                np.rot90(np.flipud(decoded), 2),
                np.rot90(np.flipud(decoded), 3),                        
            ]
        return decoded