import math
import threading
#from inputs import get_gamepad, InputEvent
import inputs
import pygame



class XboxController(object):
  WHEEL_ANGLE_CODE = 'ABS_X'
  THROTTLE_CODE = 'ABS_Z'
  BRAKE_CODE = 'ABS_RZ'

  MAX_PEDAL_VAL = math.pow(2, 8)
  MAX_WHEEL_ANGLE = math.pow(2, 15)

  def __init__(self):
    self.wheel_angle = 0
    self.throttle = 0
    self.brake = 0
    self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
    self._monitor_thread.daemon = True
    self._monitor_thread.start()


  def read(self):
    wheel_angle = self.wheel_angle
    throttle = self.throttle
    brake = self.brake
    return [wheel_angle, throttle, brake]

  def write(self, wheel_angle=None, throttle=None, brake=None):
    if wheel_angle is None or throttle is None or brake is None:
      raise ValueError("write requires wheel_angle, throttle, and brake")


    pygame.init()
    pygame.joystick.init()

    # Get the number of connected joysticks
    joystick_count = pygame.joystick.get_count()

    # Select the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Print the name of the joystick
    print(joystick.get_name())

    # Print the number of buttons on the joystick
    num_buttons = joystick.get_numbuttons()
    print(f"Number of buttons: {num_buttons}")

    # Loop through each button and print its value
    for i in range(num_buttons):
        button_value = joystick.get_button(i)
        print(f"Button {i}: {button_value}")




  def _monitor_controller(self):
    while True:
      events = inputs.get_gamepad()
      for event in events:
        #print(event.ev_type, event.code, event.state)
        if event.code == self.WHEEL_ANGLE_CODE: #wheel angle
          self.wheel_angle = event.state / XboxController.MAX_WHEEL_ANGLE # normalize between -1 and 1
        elif event.code == self.BRAKE_CODE: #brake
          self.brake = event.state / XboxController.MAX_PEDAL_VAL # normalize between 0 and 1
        elif event.code == self.THROTTLE_CODE: #throttle
          self.throttle = event.state / XboxController.MAX_PEDAL_VAL # normalize between 0 and 1