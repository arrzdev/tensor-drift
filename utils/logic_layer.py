def apply(predicted_state, speed, angle):
  steering, gas, brake = predicted_state

  #if the angle of the car is too high, counter steer
  if abs(angle) >= 55:
    steering = 1 if angle < 0 else -1

    if gas == 1 and brake == 0:
      gas = 0

  #if the car is too slow, accelerate
  elif abs(angle) <= 15 and speed <= 10 and gas == 0 and brake == 0:
    gas = 1


  #speed rule
  if speed >= 100:
    brake = 1
    gas = 0
  elif speed >= 80:
    gas = 0


  return [int(steering), int(gas), int(brake)]