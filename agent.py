import os 
import cv2
import numpy as np  
import shutil

from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

####
from utils.keyboard import key_check
from utils.screen import screenshot
from utils.wheel import XboxController
from utils.interact import prompt
from utils.model import create_model, customized_loss

#RECORDING VARS
IMAGE_SIZE = (640, 480)
SAMPLE_SIZE = (200, 66)
IMAGE_TYPE = ".png"

#MODEL VARS
INPUT_SHAPE = (66, 200, 3)
OUT_SHAPE = 3

class Agent:
  def __init__(self):
    self.path = "./pata"
    self.framesc = 0
    self.controller = XboxController()

  #### RECORD ####
  def record(self):
    # check that a dir has been specified
    if not self.path:
      print("WRONG OUTPUT DIR")
      return
    
    if os.path.exists(self.path):
      if not prompt("Directory already exists. Overwrite? (y/n)"):
        print("Aborting...")
        return
      
      shutil.rmtree(self.path, ignore_errors=True) #delete dir
  	
    #create dir
    os.mkdir(self.path)
    self.inputs = open(f"{self.path}/inputs.csv", 'a') #open file

    #record
    while True:
      self.controller.write(wheel_angle=-1, throttle=1, brake=0)
      self.img = self.get_frame()

      #apply filtering
      self.clean_frame()

      #peak_preview = cv2.resize(self.img, (1060, 650))
      self.img = cv2.resize(self.img, INPUT_SHAPE[:2]) #resize to model input shape

      # Debug line to show image
      cv2.imshow("AI Peak", self.img)
      cv2.waitKey(1)
      keys = key_check()
      #targets.append(keys)
      if keys == "H":
        break

      self.controller_data = self.controller.read()
      self.save_data()
      self.framesc += 1
    
    self.inputs.close() #close file

  def get_frame(self):
    ss = screenshot(region=(0, 20, 1920, 1040))
    return cv2.resize(ss, IMAGE_SIZE)

  def clean_frame(self):
    # define color range for green (in HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # load image and convert to HSV color space
    hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    # apply color mask to isolate green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # invert the mask to keep everything except the green pixels
    mask_inv = cv2.bitwise_not(mask)

    # apply the mask to the original image to remove the green pixels
    result = cv2.bitwise_and(self.img, self.img, mask=mask_inv)

    # apply Gaussian blur to remove noise
    result_blur = cv2.GaussianBlur(result, (3, 3), 0)

    # convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(result_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    self.img = edges

  def save_data(self):
    image_file = f"{self.path}/f_{str(self.framesc)}{IMAGE_TYPE}"
    cv2.imwrite(image_file, self.img)
    self.inputs.write(f"{image_file},{','.join(map(str, self.controller_data))}")

  #### TRAIN ####
  def train(self):
    # Load Training Data
    x_train = np.load("data/X.npy")
    y_train = np.load("data/y.npy")

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model(input_shape=INPUT_SHAPE, out_shape=OUT_SHAPE)
    
    checkpoint = ModelCheckpoint('model_weights.h3', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.compile(loss=customized_loss, optimizer=optimizers.adam())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1, callbacks=callbacks_list)