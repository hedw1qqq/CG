import numpy as np
from PIL import Image
size = 256
noise = np.random.randint(100, 200, (size, size), dtype=np.uint8)  # серый шум
Image.fromarray(noise, 'L').save('specular_noisy.png')