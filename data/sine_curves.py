import collections
from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "max_num_context", 20, "maximal number of context points. for regression)")

RegressionDescription = collections.namedtuple(
        "RegressionDescription",
        ("tr_input", "tr_output", "tr_func", "val_input", "val_output", "val_func"))

class SineCurvesReader(object):

  def __init__(self,
               batch_size,
               max_num_context,
               x_size=1,
               y_size=1,
               random_kernel_parameters=False,
               testing=False, seed=None):

      self._batch_size = batch_size
      self._max_num_context = max_num_context
      self._x_size = x_size
      self._y_size = y_size
      self._random_kernel_parameters = random_kernel_parameters
      self._testing = testing
      self._seed = seed


  def generate_curves(self):

      num_context = tf.random.uniform(
            shape=[], minval=1, maxval=self._max_num_context, dtype=tf.int32, seed=self._seed)
      if self._testing:
          num_total_points = 1000
          num_target = num_total_points - num_context
          x_values = tf.tile(
                tf.expand_dims(tf.range(-5., 5., 1. / 100, dtype=tf.float32), axis=0),
                [self._batch_size, 1])
          x_values = tf.expand_dims(x_values, axis=-1)
      else:
          num_target = tf.random_uniform(shape=(), minval=1,
                                       maxval=self._max_num_context+1 - num_context,
                                       dtype=tf.int32, seed=self._seed)
          num_total_points = num_context + num_target
          x_values = tf.random_uniform(
              [self._batch_size, num_total_points, self._x_size], -5, 5, seed=self._seed)

      amp = tf.random.uniform(shape=[self._batch_size, 1, 1], minval=0.1, maxval=5.0, seed=self._seed)
      phase = tf.random.uniform(shape=[self._batch_size, 1, 1], minval=0., maxval=np.pi, seed=self._seed)
      frequency = 1.0
      func_values = amp * tf.math.sin(frequency * (x_values - phase))

      y_values = func_values 
      # if consider observation noise: 
      # y_values = func_values + tf.random.normal(shape=tf.shape(func_values), stddev=0.5, seed=self._seed)

      if self._testing:
          
          target_x = x_values
          target_y = y_values
          target_f = func_values

          idx = tf.random_shuffle(tf.range(num_total_points), seed=self._seed)
          context_x = tf.gather(x_values, idx[:num_context], axis=1)
          context_y = tf.gather(y_values, idx[:num_context], axis=1)
          context_f = tf.gather(func_values, idx[:num_context], axis=1)
      else:
          target_x = x_values[:, num_context : num_target + num_context, :]
          target_y = y_values[:, num_context : num_target + num_context, :]
          target_f = func_values[:, num_context : num_target + num_context, :]
          context_x = x_values[:, :num_context, :]
          context_y = y_values[:, :num_context, :]
          context_f = func_values[:, :num_context, :]

      return RegressionDescription(
            tr_input=context_x,
            tr_output=context_y,
            tr_func=context_f,
            val_input=target_x,
            val_output=target_y,
            val_func=target_f)
