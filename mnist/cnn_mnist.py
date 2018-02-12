from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
CHANNEL = 1
LEARNING_RATE = 0.001

def cnn_model_fn(features, labels, mode):
  """Model function for CNN.
     This conforms to the interface expected by TensorFlow's Estimator API
     (more on this later in Create the Estimator[https://www.tensorflow.org/tutorials/layers#create_the_estimator]). 
  """
  # Input Layer
  # [batch_size, image_width, image_height, channels]
  # Note: -1 for batch size, this dimension should be dynamically computed based on the number of input values in features["x"]
  input_layer = tf.reshape(features["x"], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL])

  # Convolutional Layer #1
  # param
  #  input_layer,
  #  filters,
  #  kernel_size [width, height]
  #  padding,
  #  activation
  # this output -> [batch_size, 28, 28, 32] tensor
  conv1 = tf.layers.conv2d(input_layer, 32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

  # Pooling Layer #1
  # param
  #  input_layer,
  #  pool_size [width, height],
  #  strides
  # this output -> [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)

  # Convolutional Layer #2
  # this output -> [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(pool1, 64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
  # Pooling Layer #2
  # this output -> [batch_size, 7, 7, 32]
  pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)

  # Dense Layer
  # beforehand flatten our feature map (pool2) to shape [batch_size, features] 
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  # Dropout
  # this output -> [batch_size, 1024]
  # The training argument takes a boolean specifying whether or not the model is currently being run in training mode;
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  
  # Logits Layer
  # this output -> [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  # predictions
  #  argmas: axis argument specifies the axis of the input tensor along which to find the greatest value. <- [batch_size, 10]
  #  
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  # return an EstimatorSpec object:
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  else: #(both TRAIN and EVAL modes)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # For multiclass classification problems like MNIST, cross entropy is typically used as the loss metric.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #instead of
    '''
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    '''

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    else: 
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def main(unused_argv):

  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train: the raw pixel values for 55,000 images of hand-drawn digits
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval: 10,000 images
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="./mnist_convnet_model")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  # num_epochs=None means that the model will train until the specified number of steps is reached
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=10,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()

