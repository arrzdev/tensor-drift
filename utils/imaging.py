"""from skimage.transform import resize
import numpy as np

def resize_image(img, sample):
  img = np.array(img)
  im = resize(img, tuple(sample.values()))
  im_arr = im.reshape(tuple(sample.values()))
  return im_arr
"""

import cv2

def resize_image(img, sample):
  h, w = sample["img_h"], sample["img_w"]
  im = cv2.resize(img, (w, h))
  return im

def color_channel(image, cix, value):
  height, width, _ = image.shape

  for y in range(height):
    for x in range(width):
      image[y, x, cix] = value
  
  return image