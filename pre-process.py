import os
import numpy as np
import cv2

def clean_image(frame):
  # define color range for green (in HSV)
  lower_green = np.array([40, 40, 40])
  upper_green = np.array([70, 255, 255])

  # load image and convert to HSV color space
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # apply color mask to isolate green pixels
  mask = cv2.inRange(hsv, lower_green, upper_green)

  # invert the mask to keep everything except the green pixels
  mask_inv = cv2.bitwise_not(mask)

  # apply the mask to the original image to remove the green pixels
  result = cv2.bitwise_and(frame, frame, mask=mask_inv)

  # apply Gaussian blur to remove noise
  result_blur = cv2.GaussianBlur(result, (3, 3), 0)

  # convert to grayscale and apply Canny edge detection
  gray = cv2.cvtColor(result_blur, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, threshold1=50, threshold2=150)

  # convert the 2D grayscale image to a 3D image with 3 channels
  edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

  return edges


# Set image directory and extension
img_dir = './pata'
img_ext = '.png'

# Get list of image files
img_files = [f for f in os.listdir(img_dir) if f.endswith(img_ext)]

# Preprocess images and store in X array
X = np.zeros((len(img_files), 66, 200, 3), dtype=np.float32)
for i, img_file in enumerate(img_files):
  print(f"Processing image {i+1}/{len(img_files)}: {img_file}")
  img = clean_image(cv2.imread(os.path.join(img_dir, img_file)))
  img = cv2.resize(img, (200, 66))
  img = np.array(img, dtype=np.float32)
  img /= 255.0
  X[i] = img

# Save preprocessed images
np.save('X.npy', X)