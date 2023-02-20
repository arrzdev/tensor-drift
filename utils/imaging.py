from skimage.transform import resize
import numpy as np

def resize_image(img, sample):
  img = np.array(img)
  im = resize(img, tuple(sample.values()))
  im_arr = im.reshape(tuple(sample.values()))
  return im_arr