import threading
import win32api
import win32con

import ctypes

#https://stackoverflow.com/questions/56777292/how-to-use-python-to-control-a-game
SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
  _fields_ = [
    ("wVk", ctypes.c_ushort),
    ("wScan", ctypes.c_ushort),
    ("dwFlags", ctypes.c_ulong),
    ("time", ctypes.c_ulong),
    ("dwExtraInfo", PUL)
  ]

class HardwareInput(ctypes.Structure):
  _fields_ = [
    ("uMsg", ctypes.c_ulong),
    ("wParamL", ctypes.c_short),
    ("wParamH", ctypes.c_ushort)
  ]

class MouseInput(ctypes.Structure):
  _fields_ = [
    ("dx", ctypes.c_long),
    ("dy", ctypes.c_long),
    ("mouseData", ctypes.c_ulong),
    ("dwFlags", ctypes.c_ulong),
    ("time",ctypes.c_ulong),
    ("dwExtraInfo", PUL)
  ]

class Input_I(ctypes.Union):
  _fields_ = [
    ("ki", KeyBdInput),
    ("mi", MouseInput),
    ("hi", HardwareInput)
  ]

class Input(ctypes.Structure):
  _fields_ = [
    ("type", ctypes.c_ulong),
    ("ii", Input_I)
  ]

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def read_keys(fkeys):
  keys = []
  for fkey in fkeys:
    if win32api.GetAsyncKeyState(fkey):
      keys.append(fkey)
  return keys

class KeyboardController():
  def __init__(self):
    self.sterring = 0 # -1 or 1
    self.throttle = 0 # 0 or 1
    self.brake = 0 # 0 or 1

    #filtered keys
    self.keyboard_mapping = {
      "throttle": (win32con.VK_UP, 0xC8),
      "brake": (win32con.VK_DOWN, 0xD0),
      "wheel_left": (win32con.VK_LEFT, 0xCB),
      "wheel_right": (win32con.VK_RIGHT, 0xCD)
    }

    self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
    self._monitor_thread.daemon = True
    self._monitor_thread.start()

  """
  returns the current state of the physical controller
  """
  def read(self):
    return [self.sterring, self.throttle, self.brake]

  """
  injector of keystrokes
  """
  def write(self, action=None, key=None):
    print("writting", action, key)
    if action == "press":
      PressKey(self.keyboard_mapping[key][1])
    elif action == "release":
      ReleaseKey(self.keyboard_mapping[key][1])

  """
  listener on physical keyboard
  """
  def _monitor_controller(self):
    filter_keys = [v[0] for v in self.keyboard_mapping.values()]
    while True:
      event_keys = read_keys(filter_keys)
      self.throttle = 1 if self.keyboard_mapping["throttle"][0] in event_keys else 0
      self.brake = 1 if self.keyboard_mapping["brake"][0] in event_keys else 0
      self.sterring = -1 if self.keyboard_mapping["wheel_left"][0] in event_keys else 1 if self.keyboard_mapping["wheel_right"][0] in event_keys else 0

