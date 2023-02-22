import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#https://stackoverflow.com/questions/26090597/why-pytesseract-does-not-recognise-single-digits
def get_game_data(image, speed_grid, angle_grid, angle_side_grid):
  gry1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  (h, w) = gry1.shape[:2]
  upscaled = cv2.resize(gry1, (w*2, h*2))

  values = []
  for v in [speed_grid, angle_grid]:
    grid = upscaled[v["y"]:v["y"]+v["h"], v["x"]:v["x"]+v["w"]]
    thr = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    result = pytesseract.image_to_string(thr, config="--psm 6 digits")

    #normalize result
    normalizations = {
      "O": "0",
      "o": "0",
      "I": "1",
      "l": "1",
      "S": "5",
      "s": "5",
      " ": ""
    }

    result = result.lstrip("0")
    for k, v in normalizations.items():
      result = result.replace(k, v)

    try:
      values.append(int(result))
    except:
      values.append(0)
  
  #get angle side
  if values[1] > 0:
    values[1] *= get_angle_side(image, angle_side_grid)

  return values

def get_angle_side(image, angle_side_grid):
  upscaled = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))

  gry1 = upscaled[angle_side_grid["y"]:angle_side_grid["y"]+angle_side_grid["h"], angle_side_grid["x"]:angle_side_grid["x"]+angle_side_grid["w"]]

  # Convert BGR to HSV
  hsv = cv2.cvtColor(gry1, cv2.COLOR_BGR2HSV)
  # define range of blue color in HSV
  lower_blue = np.array([0,100,0])
  upper_blue = np.array([255,255,255])

  # Threshold the HSV image to get only blue colors
  mask = cv2.inRange(hsv, lower_blue, upper_blue)

  return -1 if mask.any() else 1
