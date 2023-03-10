import os 
import numpy as np  
import shutil
import time
import cv2
import win32gui
import time
from PIL import ImageGrab

from keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

####
from utils.screen import screenshot
from utils.pytesseract import get_game_data
from utils.imaging import resize_image
from utils.wheel import WheelController
from utils.keyboard import KeyboardController, read_keys 
from utils.interact import prompt
from utils.model import create_model, customized_loss
from utils.emulator import EmulatorEngine
from utils import logic_layer

###
from skimage.io import imread


#RECORDING VARS
IMAGE_TYPE = ".png"

#MODEL VARS
SAMPLE = {
  "img_h": 66,
  "img_w": 200,
  "img_d": 3
}

INPUT = 5
OUTPUT = 3

class Agent:
  def __init__(self):   
    self.sample_path = "./samples"
    self.data_path = "./data"
    self.game_region = (0, 25, 1920, 1030)
    self.model_path = "./model_weights.h5"

    self.framesc = 0
    self.recording = False

    self.wheel_controller = WheelController()
    self.keyboard_controller = KeyboardController()
    self.emulator = EmulatorEngine(self.keyboard_controller) #receives controller

    self.keyboard_controller_data = [0, 0, 0]

    #speed grid
    self.speed_grid = {
      "x": 3500,
      "y": 1625,
      "w": 200,
      "h": 100
    }

    self.angle_grid = {
      "x": 1850,
      "y": 1825,
      "w": 125,
      "h": 115
    }

    self.angle_side_grid = {
      "x": 3100, #3100 #3020
      "y": 1720,
      "w": 70, #70
      "h": 10
    }

### RECORDING ###
  def record(self):
    # check that a dir has been specified
    if not self.sample_path:
      print("WRONG OUTPUT DIR")
      return

    if os.path.exists(self.sample_path):
        if not prompt("Directory already exists. Overwrite? (y/n)"):
          #continue from last frame
          for file in os.listdir(self.sample_path):
            if file.endswith(IMAGE_TYPE):
              frame_index = int(file.split("_")[-1].split(".")[0])
              if frame_index >= self.framesc:
                self.framesc = frame_index + 1
          if self.framesc > 0:
            print(f"Continuing from frame {self.framesc}")
        else:
          #delete dir
          shutil.rmtree(self.sample_path, ignore_errors=True)
          #create dir
          os.mkdir(self.sample_path)
    else:
      #create dir
      os.mkdir(self.sample_path)

    # open file
    inputs_path = f"{self.sample_path}/inputs.csv"
    with open(inputs_path, "a") as self.inputs:
      # wait for H to start recording
      print("Press H to start recording.")
      while True:
        keys = read_keys([ord("H")])
        if keys:
          if self.recording:  # already recording
            print("Recording stopped.")
            self.recording = False
            return
          else:  # start recording
            print("Starting recording in 3 seconds...")
            time.sleep(3)
            print("Recording started.")
            self.recording = True

        if self.recording:
          self.screenshot = screenshot(self.game_region)
          self.keyboard_controller_data = self.keyboard_controller.read() # read controller data
          print(self.keyboard_controller_data)
          if not self.save_data():
            print("Error saving data")
            return
          # use image object for model
          self.framesc += 1
  
  def save_data(self):
    image_file = f"{self.sample_path}/img_{self.framesc}{IMAGE_TYPE}"
    saved = cv2.imwrite(image_file, self.screenshot)

    if saved:
      # write csv line
      self.inputs.write(image_file + ',' + ','.join(map(str, self.keyboard_controller_data)) + '\n')

    return saved

  #### PREPROCESS ####
  def pre_process(self):
    print("Preparing data")
    image_files = np.loadtxt(f"{self.sample_path}/inputs.csv", delimiter=',', dtype=str, usecols=(0,))

    #count samples
    num_samples = len(image_files)
    print(f"Found {num_samples} samples")

    try:
      X = np.load(f"{self.data_path}/X.npy")
      num_processed = X.shape[::][0]
      X.resize((num_samples, SAMPLE["img_h"], SAMPLE["img_w"], SAMPLE["img_d"]))
    except:
      X = np.empty(shape=(num_samples, SAMPLE["img_h"], SAMPLE["img_w"], SAMPLE["img_d"]), dtype=np.float32)
      num_processed = 0

    y = np.loadtxt(f"{self.sample_path}/inputs.csv", delimiter=',', usecols=(1,2,3)) 

    #process the new images starting on img_{X.shape[0]}
    for i in range(num_processed, num_samples):
      print(f"Processing image {i+1}/{num_samples}")
      image = imread(image_files[i])
      speed, angle = get_game_data(image, self.speed_grid, self.angle_grid, self.angle_side_grid)

      #normalize speed and angle to match rgb color
      nspeed = (speed/200)*255
      nangle = (abs(angle)/180)*255

      #color the image using cv2 and rgb values
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      #add speed as R channel since speed is 
      image = np.stack((image,)*3, axis=-1)
      image[:,:,0] = nspeed

      #decide in which channel will the angle be encoded based on the side of the car
      if angle <= 0:
        image[:,:,2] = nangle
      else:
        image[:,:,1] = nangle

      image = resize_image(image, SAMPLE)
      X[i] = image

    #save the new arrays
    np.save(f"{self.data_path}/X.npy", X)
    np.save(f"{self.data_path}/y.npy", y)

    print("Data prepared")

  def train(self):
    print("STARTING TRAINING")
    # Load Training Data
    x_train = np.load(f"{self.data_path}/X.npy")
    y_train = np.load(f"{self.data_path}/y.npy")

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    #check if model exists
    if os.path.exists(self.model_path):
      if not prompt("Model already exists. Overwrite? (y/n)"):
        self.model = load_model(self.model_path, custom_objects={'customized_loss': customized_loss})
        #continue from last frame
        print("Continuing from last model")
      else:
        #delete dir
        shutil.rmtree(self.model_path, ignore_errors=True)
        #create dir
        print("Creating new model")
        self.model = create_model(SAMPLE, OUTPUT)
    else:
      #create dir
      print("Creating new model")
      self.model = create_model(sinput=tuple(SAMPLE.values()), soutput=OUTPUT)

    #set up callbacks 
    checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    self.model.compile(loss=customized_loss, optimizer=optimizers.Adam())
    self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1, callbacks=[checkpoint])


  #### PLAY ####
  def play(self):
    # Load the trained model
    model = create_model(keep_prob=1, sinput=tuple(SAMPLE.values()), soutput=OUTPUT)
    model.load_weights(self.model_path)

    # Play the game
    while True:
      if any(self.wheel_controller.read()):
        #print("User interacting with the wheel. Ignoring AI prediction.")
        self.emulator.sync_controller()
        continue

      # capture the game screen
      ss = screenshot(self.game_region)

      speed, angle = get_game_data(ss, self.speed_grid, self.angle_grid, self.angle_side_grid)

      #normalize speed and angle to match rgb color
      nspeed = (speed/200)*255
      nangle = (abs(angle)/180)*255

      #color the image using cv2 and rgb values
      image = cv2.cvtColor(ss, cv2.COLOR_BGR2GRAY)

      #add speed as R channel since speed is 
      image = np.stack((image,)*3, axis=-1)
      image[:,:,0] = nspeed

      #decide in which channel will the angle be encoded based on the side of the car
      if angle <= 0:
        image[:,:,2] = nangle
      else:
        image[:,:,1] = nangle

      # preprocess the screpenshot
      image = resize_image(image, SAMPLE)

      # predict the action to take
      model_input = np.expand_dims(image, axis=0)

      # Use the model to predict the next state
      action = model.predict(model_input, batch_size=1)

      # Round the first value in the action array to the nearest integer
      action[0][0] = np.round(action[0][0])

      # Limit the values of the other two elements to 0 or 1
      action[0][1:] = np.clip(np.round(action[0][1:]), 0, 1)

      # Scale the first value to be either -1, 0 or 1
      action[0][0] = np.sign(action[0][0])

      action = action[0]

      print(action)

      logic_action = logic_layer.apply(action, speed, angle)
      print(logic_action)
      
      # Send the predicted action to the emulator
      self.emulator.step(logic_action)
