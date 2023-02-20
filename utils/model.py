from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import backend as K
import tensorflow as tf

def create_model(keep_prob=0.8 , sinput=None, soutput=None):
  model = Sequential()

  # NVIDIA's model
  model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=sinput))
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
  model.add(Dense(soutput, activation='softsign'))

  return model

def customized_loss(y_true, y_pred, loss='euclidean'):
  # Simply a mean squared error that penalizes large joystick summed values
  if loss == 'L2':
    L2_norm_cost = 0.001
    y_true_float = K.cast(y_true, dtype=tf.float32)
    val = K.mean(K.square((y_pred - y_true_float)), axis=-1) \
      + K.sum(K.square(y_pred), axis=-1)/2 * L2_norm_cost
  # euclidean distance loss
  elif loss == 'euclidean':
    y_true_float = K.cast(y_true, dtype=tf.float32)
    val = K.sqrt(K.sum(K.square(y_pred-y_true_float), axis=-1))
  elif loss == 'experimental':
    # Set alpha to control the importance of larger deviations
    alpha = 0.5
    # Compute the mean squared error between true and predicted values
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    # Compute the weight for each sample based on the absolute error
    weights = tf.math.abs(y_true - y_pred) * alpha + 1.0
    # Compute the weighted mean squared error
    val = tf.reduce_mean(tf.multiply(mse, weights), axis=-1)
  return val
