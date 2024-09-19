import numpy as np

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
