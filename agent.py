import os 
import cv2
from PIL import Image
import numpy as np  
import shutil
import random
import time

from keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

####
from utils.screen import screenshot
from utils.wheel import WheelController
from utils.keyboard import KeyboardController, read_keys 
from utils.interact import prompt
from utils.model import create_model, customized_loss
from utils.emulator import EmulatorEngine

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

    self.frames_info = []
    self.wheel_controller = WheelController()
    self.keyboard_controller = KeyboardController()
    self.emulator = EmulatorEngine(self.keyboard_controller) #receives controller

    self.keyboard_controller_data = [0, 0, 0]

  #### RECORD ####
  def record(self):
    # check that a dir has been specified
    if not self.path:
      print("WRONG OUTPUT DIR")
      return
    
    if os.path.exists(self.path):
      if not prompt("Directory already exists. Overwrite? (y/n)"):
        #continue from last frame
        for file in os.listdir(self.path):
          if file.endswith(IMAGE_TYPE):
            frame_index = int(file.split("_")[-1].split(".")[0])
            if frame_index >= self.framesc:
              self.framesc = frame_index + 1
        if self.framesc > 0:
          print(f"Continuing from frame {self.framesc}")

        self.frames_info = np.load(f"{self.path}/y.npy")
      else:
        #delete dir
        shutil.rmtree(self.path, ignore_errors=True)
        #create dir
        os.mkdir(self.path)
    else:
      #create dir
      os.mkdir(self.path)
    
    #open file
    #self.inputs = open(f"{self.path}/inputs.csv", 'a') #open file

    #record
    #wait for H to start recording
    print("Press H to start recording")
    recording = False
    while True:
      self.get_frame()

      # Convert PIL Image to NumPy array
      frame_np = np.array(self.frame)
      # Convert RGB to BGR
      frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
      frame_np = cv2.resize(frame_np, (1060, 650))

      cv2.imshow("AI Peak", frame_np)
      cv2.waitKey(1)

      if recording: #already recording
        #stop recording condition
        if len(read_keys([ord("H")])):
          break

        self.keyboard_controller_data = self.keyboard_controller.read()
        print(self.keyboard_controller_data)
        self.save_data()
        self.framesc += 1
      
      elif len(read_keys([ord("H")])): #start recording
        recording = True
        print("Starting recording in 3 seconds...")
        time.sleep(3)
        print("Recording started")
        continue

    print("Recording stopped")
    np.save(f"{self.path}/y.npy", self.frames_info)
    #self.inputs.close() #close file

  def get_frame(self):
    ss = screenshot(region=(0, 20, 1920, 1040))
    self.frame = Image.fromarray(cv2.cvtColor(ss, cv2.COLOR_BGR2RGB))

  def clean_frame(self):
    # define color range for green (in HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # load image and convert to HSV color space
    hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

    # apply color mask to isolate green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # invert the mask to keep everything except the green pixels
    mask_inv = cv2.bitwise_not(mask)

    # apply the mask to the original image to remove the green pixels
    result = cv2.bitwise_and(self.frame, self.frame, mask=mask_inv)

    # apply Gaussian blur to remove noise
    result_blur = cv2.GaussianBlur(result, (3, 3), 0)

    # convert to grayscale and apply Canny edge detection
    gray = cv2.cvtColor(result_blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    self.frame = edges

  def resize_frame(self):
    # Resize the image to the specified dimensions
    im = cv2.resize(self.frame, INPUT_SHAPE[:2], interpolation=cv2.INTER_AREA)
    # Convert the resized image to an array and reshape it
    self.frame = im.reshape(INPUT_SHAPE)

  def save_data(self):
    image_file = f"{self.path}/f_{str(self.framesc)}{IMAGE_TYPE}"

    # Convert PIL Image to NumPy array
    frame_np = np.array(self.frame)
    # Convert RGB to BGR
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(image_file, frame_np)
    #self.inputs.write(f"{image_file},{','.join(map(str, self.keyboard_controller_data))}")
    self.frames_info.append(self.keyboard_controller_data)

  #### TRAIN ####
  def train(self):
    # Load Training Data
    x_train = np.load("./X.npy")
    y_train = np.load("pata/y.npy")

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    model = create_model(input_shape=INPUT_SHAPE, output_shape=OUT_SHAPE)
    
    checkpoint = ModelCheckpoint('model_weights.h3', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.compile(loss=customized_loss, optimizer=optimizers.Adam())
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1, callbacks=callbacks_list)

  #### PLAY ####
  def play(self):
    self.model = create_model(keep_prob=1, input_shape=INPUT_SHAPE, output_shape=OUT_SHAPE)
    self.model.load_weights("model_weights.h3")

    print("Press H to start playing")
    while not len(read_keys([ord("H")])):
      continue

    print("Starting playing in 3 seconds...")
    time.sleep(3)

    flag = False
    while not len(read_keys([ord("H")])):
      #get physical controller state
      if any(self.wheel_controller.read()):
        if not flag:
          flag = True
          print("Physical input detected")

        self.emulator.sync_controller()
        time.sleep(0.1)
        continue
      flag = False

      #make prediction
      self.get_frame()
      self.resize_frame()

      vec = np.expand_dims(self.frame, axis=0)
      self.prediction = self.model.predict(vec, batch_size=1)

      #apply prediction to virtual controller
      #use random prediction for now

      print("Ai input detected: ", self.prediction)
      self.emulator.step(self.prediction)