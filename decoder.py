import gzip
import json

import numpy as np


class Decoder:
    def __init__(self, transforms_path, image_size=None, iterations=20):
        self.transforms_path = transforms_path
        self.iterations = iterations
        if transforms_path.endswith('.gz'):
            with gzip.open(transforms_path, 'rt', encoding='utf-8') as f:
                self._doc = json.load(f)
        else:
            with open(transforms_path, encoding='utf-8') as f:
                self._doc = json.load(f)

        if self._doc.get("format") != "fractal-pifs" or "transforms" not in self._doc:
            raise ValueError(
                "Unrecognized transform file. Expected a JSON file from this encoder (format: fractal-pifs)."
            )

        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = (
                int(self._doc["image_height"]),
                int(self._doc["image_width"]),
            )

        self.transforms = self._doc["transforms"]

    def decode(self, on_iteration=None):
        """Decode the fractal transform iteratively.

        Parameters
        ----------
        on_iteration:
            Optional callable invoked after each image state is produced,
            including the initial random seed (iteration 0, before any
            transforms are applied).  Signature::

                on_iteration(iteration: int, image: np.ndarray) -> None

            ``iteration`` is 0 for the seed image, then 1 … N for each
            successive transform application.  ``image`` is a read-only view
            of the current float32 array in [0, 1].  The callback is called
            synchronously before the next iteration begins, so it is safe to
            copy or save the array inside the callback.

        Returns
        -------
        np.ndarray
            The final decoded image (same array that was passed to the last
            ``on_iteration`` call).
        """
        decoded = np.random.rand(*self.image_size)

        if on_iteration is not None:
            decoded.flags.writeable = False
            on_iteration(0, decoded)
            decoded.flags.writeable = True

        domain_images = self._make_domain_images(decoded)

        for i in range(self.iterations):
            print(f"Decoding iteration {i + 1}/{self.iterations}")
            new_decoded = np.zeros_like(decoded)
            for t in self.transforms:
                domain_source_index = int(t["domain_source_index"])
                domain_x = int(t["domain_x"])
                domain_y = int(t["domain_y"])
                domain_size = int(t["domain_size"])
                domain_scale = int(t["domain_scale"])
                range_x = int(t["range_x"])
                range_y = int(t["range_y"])
                range_size = int(t["range_size"])
                scale = float(t["scale"])
                bias = float(t["bias"])

                domain_block = domain_images[domain_source_index][
                    domain_y : domain_y + domain_size, domain_x : domain_x + domain_size
                ]
                if domain_scale > 1:
                    domain_block = domain_block.reshape(
                        domain_size // domain_scale,
                        domain_scale,
                        domain_size // domain_scale,
                        domain_scale,
                    ).mean(axis=(1, 3))
                range_block = domain_block * scale + bias
                new_decoded[range_y : range_y + range_size, range_x : range_x + range_size] = range_block

            decoded = new_decoded
            domain_images = self._make_domain_images(decoded)

            if on_iteration is not None:
                decoded.flags.writeable = False
                on_iteration(i + 1, decoded)
                decoded.flags.writeable = True

        return decoded

    @staticmethod
    def _make_domain_images(image):
        return [
            image,
            np.rot90(image),
            np.rot90(image, 2),
            np.rot90(image, 3),
            np.flipud(image),
            np.rot90(np.flipud(image)),
            np.rot90(np.flipud(image), 2),
            np.rot90(np.flipud(image), 3),
        ]
