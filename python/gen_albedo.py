import numpy as np
from PIL import Image
size = 256
img = np.zeros((size, size), dtype=np.uint8)
img[size//2-8:size//2+8, :] = 255  # горизонтальная линия
img[:, size//2-8:size//2+8] = 255  # вертикальная
Image.fromarray(img, 'L').save('specular_cross.png')