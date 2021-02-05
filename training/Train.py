import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import onnxruntime as rt
import keras2onnx
import numpy as np
import gzip
import yaml
import json
import logging

logger = logging.getLogger(__name__)

def train(lol="lal", epochs=1, metadata_file=r"./outputs/training_metadata.yaml", learning_rate=0.001):
  print(__name__)
  logger.info(lol)

  def load_data(path, num_images=5, image_size=28):
    f = gzip.open("./data/" + path,'r')
    logger.info(path)
    if image_size < 28:
      f.read(8)
    else:
      f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    logger.info(data.shape)
    data = data.reshape(num_images, image_size, image_size)
    return data


  data_paths = os.listdir("./data")
  logger.info(data_paths)
  test_paths = list(filter(lambda x: "t10k" in x , data_paths))
  test_images_path = list(filter(lambda x: "images" in x , test_paths))[0]
  test_labels_path = list(filter(lambda x: "labels" in x , test_paths))[0]
  train_paths = list(filter(lambda x: "train" in x , data_paths))
  train_images_path = list(filter(lambda x: "images" in x , train_paths))[0]
  train_labels_path = list(filter(lambda x: "labels" in x , train_paths))[0]

  (x_train, y_train), (x_test, y_test) = (load_data(train_images_path, 60_000), load_data(train_labels_path, 60_000, 1)), (load_data(test_images_path, 10_000), load_data(test_labels_path, 10_000, 1))

  x_train, x_test = x_train / 255.0, x_test / 255.0

  # Add a channels dimension
  x_train = x_train[..., tf.newaxis].astype("float32")
  x_test = x_test[..., tf.newaxis].astype("float32")
  logger.info(x_train.shape)
  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32)

  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

  class MyModel(Model):
    def __init__(self):
      super(MyModel, self).__init__()
      self.conv1 = Conv2D(32, 3, activation='relu')
      self.flatten = Flatten()
      self.d1 = Dense(128, activation='relu')
      self.d2 = Dense(10)

    def call(self, x):
      x = self.conv1(x)
      x = self.flatten(x)
      x = self.d1(x)
      return self.d2(x)

  # Create an instance of the model
  model = MyModel()

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  model.compile(optimizer=optimizer,
                loss=train_loss,
                metrics=[train_accuracy])
  @tf.function
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=True)
      loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

  @tf.function
  def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

  EPOCHS = epochs

  for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
      train_step(images, labels)

    for test_images, test_labels in test_ds:
      test_step(test_images, test_labels)

    logger.info(json.dumps({
      "Epoch": epoch + 1,
      "Loss": train_loss.result().numpy().item(),
      "Accuracy":train_accuracy.result().numpy().item() * 100,
      "Test Loss": test_loss.result().numpy().item(),
      "Test Accuracy": test_accuracy.result().numpy().item() * 100
    })
    )
  print("saving???")
  
  model.save("outputs/mnist", save_format='tf')
  # onnx_model = keras2onnx.convert_keras(model, "mnist")

  # temp_model_file = './outputs/model.onnx'
  # keras2onnx.save_model(onnx_model, temp_model_file)

  metadata = {
    'training_statistics' : {
      'train_accuracy' : train_accuracy.result().numpy().item(),
      'test_accuracy' : test_accuracy.result().numpy().item(),
      'train_loss' : train_loss.result().numpy().item(),
      'test_loss' : test_loss.result().numpy().item(),
      'Epochs' : EPOCHS
      }
    }
  logger.info((metadata))

  with open(metadata_file, 'w') as file:
      documents = yaml.dump(metadata, file)

  logger.info("trying shit")
  # session = rt.InferenceSession(temp_model_file)
  # input_name = session.get_inputs()[0].name
  # output_name = session.get_outputs()[0].name


