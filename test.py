from PIL import Image
import io
import numpy as np
f = io.BytesIO(img)
np.set_printoptions(threshold=np.nan)
im = Image.open(f)
print(im.mode)
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    print(im_width * im_height)
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 4)).astype(np.uint8)

image_np = load_image_into_numpy_array(im)
