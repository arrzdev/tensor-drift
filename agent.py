import os 
from PIL import Image
import numpy as np  
import shutil
import time

#idk
import mss

from keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

####
#from utils.screen import screenshot
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

    self.screenshot_manager = mss.mss()
    self.framesc = 0

    self.wheel_controller = WheelController()
    self.keyboard_controller = KeyboardController()
    self.emulator = EmulatorEngine(self.keyboard_controller) #receives controller

    self.keyboard_controller_data = [0, 0, 0]

  #### RECORD ####
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
    
    #open file
    self.inputs = open(f"{self.sample_path}/inputs.csv", 'a') #open file

    #record
    #wait for H to start recording
    print("Press H to start recording")
    recording = False
    while True:
      self.take_screenshot() #take screenshot

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
    self.inputs.close() #close file

  def take_screenshot(self):
    # Get raw pixels from the screen
    sct_img = self.screenshot_manager.grab({
      "top": 30,
      "left": 0,
      "width": 1920,
      "height": 1000
    })

    # Create the Image
    self.screenshot = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
  
  def save_data(self):
    image_file = self.sample_path+'/'+'img_'+str(self.framesc)+IMAGE_TYPE
    self.screenshot.save(image_file) 
    
    # write csv line
    self.inputs.write(image_file + ',' + ','.join(map(str, self.keyboard_controller_data)) + '\n')

  #### PREPROCESS ####
  def pre_process(self):
    print("Preparing data")
    image_files = np.loadtxt(f"{self.sample_path}/inputs.csv", delimiter=',', dtype=str, usecols=(0,))

    #count samples
    num_samples = len(image_files)
    print(f"Found {num_samples} samples")

    #init arrays
    y = np.loadtxt(f"{self.sample_path}/inputs.csv", delimiter=',', usecols=(1,2,3)) 
    X = np.empty(shape=(num_samples, SAMPLE["img_h"], SAMPLE["img_w"], OUTPUT), dtype=np.float32)

    #prepare input data (images)
    for idx, image_file in enumerate(image_files):
      print(f"Processing image {idx+1}/{num_samples}")
      image = imread(image_file)
      vec = resize_image(image, SAMPLE)
      X[idx] = vec

    print("Saving to file...")

    np.save(f"{self.data_path}/X.npy", X)
    np.save(f"{self.data_path}/y.npy", y)

    print("Done!")

  def train(self):
    print("STARTING TRAINING")
    # Load Training Data
    x_train = np.load(f"{self.data_path}/X.npy")
    y_train = np.load(f"{self.data_path}/y.npy")

    print(x_train.shape[0], 'train samples')

    # Training loop variables
    epochs = 100
    batch_size = 50

    self.model = create_model(sinput=tuple(SAMPLE.values()), soutput=OUTPUT)
    
    checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    self.model.compile(loss=customized_loss, optimizer=optimizers.Adam())
    self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1, callbacks=callbacks_list)

  #### PLAY ####
  def play(self):
    # Load the trained model
    model = create_model(keep_prob=1, sinput=tuple(SAMPLE.values()), soutput=OUTPUT)
    model.load_weights('model_weights.h5')

    # Play the game
    while True:
      # Get the current state
      self.take_screenshot()
      state = resize_image(self.screenshot, SAMPLE)
      state = np.expand_dims(state, axis=0)

      # Use the model to predict the next state
      action = model.predict(state, batch_size=1)

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
