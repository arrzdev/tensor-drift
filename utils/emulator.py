from utils.keyboard import KeyboardController

class EmulatorEngine():
  def __init__(self, controller: KeyboardController):
    self.controller = controller
    self.state = [0, 0, 0]

  def step(self, state):
    rstate = [0 if state[i] == self.state[i] else self.state[i] for i in range(len(self.state))]
    pstate = [0 if state[i] == self.state[i] else state[i] for i in range(len(self.state))]
  
    self.edit_state("release", rstate)
    self.edit_state("press", pstate)

    self.state = state
          
  def sync_controller(self):
    self.edit_state("release", self.state)
    self.state = [0,0,0]    
          
  def edit_state(self, mode, state):
    #steering
    if state[0] == 1:
      self.controller.write(action=mode, key="wheel_right")
    elif state[0] == -1:
      self.controller.write(action=mode, key="wheel_left")

    #throttle
    if state[1] == 1:
      self.controller.write(action=mode, key="throttle")

    #brake
    if state[2] == 1:
      self.controller.write(action=mode, key="brake")