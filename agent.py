import os 
from PIL import Image
import numpy as np  
import shutil
import time
import cv2
import win32api
import win32con
import win32gui
import time
from PIL import ImageGrab

#idk
import mss

from keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

####
from utils.screen import screenshot
from utils.imaging import resize_image
from utils.wheel import WheelController
from utils.keyboard import KeyboardController, read_keys 
from utils.interact import prompt
from utils.model import create_model, customized_loss
from utils.emulator import EmulatorEngine

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

OUTPUT = 3

class Agent:
  def __init__(self):   
    self.sample_path = "./samples"
    self.data_path = "./data"
    self.game_region = (0, 25, 1920, 1030)
    self.model_path = "./model_weights.h3"

    self.screenshot_manager = mss.mss()
    self.framesc = 0
    self.recording = False

    self.wheel_controller = WheelController()
    self.keyboard_controller = KeyboardController()
    self.emulator = EmulatorEngine(self.keyboard_controller) #receives controller

    self.keyboard_controller_data = [0, 0, 0]

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

  def take_screenshot(self):
    hwnd = win32gui.GetForegroundWindow()
    rect = win32gui.GetWindowRect(hwnd)
    x, y, width, height = rect
    x += 10  # Remove the left bar
    y += 32  # Increase the y value to remove the top bar
    width -= 16
    height -= 50  # Decrease the height to remove the top bar and bottom taskbar
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    return screenshot
  
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
      X.resize((num_samples, SAMPLE["img_h"], SAMPLE["img_w"], OUTPUT))
    except:
      X = np.empty(shape=(num_samples, SAMPLE["img_h"], SAMPLE["img_w"], OUTPUT), dtype=np.float32)
      num_processed = 0

    y = np.loadtxt(f"{self.sample_path}/inputs.csv", delimiter=',', usecols=(1,2,3)) 

    #process the new images starting on img_{X.shape[0]}
    for i in range(num_processed, num_samples):
      print(f"Processing image {i+1}/{num_samples}")
      image = imread(image_files[i])
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
        print("User interacting with the wheel. Ignoring AI prediction.")
        self.emulator.sync_controller()
        continue

      # capture the game screen
      ss = screenshot(self.game_region)

      # preprocess the screenshot
      ss = resize_image(ss, SAMPLE)

      # predict the action to take
      model_input = np.expand_dims(ss, axis=0)

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

      # Send the predicted action to the emulator
      self.emulator.step(action)
