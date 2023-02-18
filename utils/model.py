from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as K

def create_model(keep_prob = 0.8 , input_shape = None, output_shape = None):
  if input_shape is None or output_shape is None:
    raise ValueError("Model input and output shapes must be specified")

  model = Sequential()

  # NVIDIA's model
  model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
  model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(1164, activation='relu'))
  drop_out = 1 - keep_prob
  model.add(Dropout(drop_out))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(drop_out))
  model.add(Dense(50, activation='relu'))
  model.add(Dropout(drop_out))
  model.add(Dense(10, activation='relu'))
  model.add(Dropout(drop_out))
  model.add(Dense(output_shape, activation='softsign'))

  return model

def customized_loss(y_true, y_pred, loss='euclidean'):
  # Simply a mean squared error that penalizes large joystick summed values
  if loss == 'L2':
    L2_norm_cost = 0.001
    val = K.mean(K.square((y_pred - y_true)), axis=-1) \
      + K.sum(K.square(y_pred), axis=-1)/2 * L2_norm_cost
  # euclidean distance loss
  elif loss == 'euclidean':
    val = K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))
  return val