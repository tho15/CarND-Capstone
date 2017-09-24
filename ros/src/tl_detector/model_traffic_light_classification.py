import numpy as np
import os
import simplejson

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img

np.random.seed(1984)
K.set_image_dim_ordering("tf")


def get_model(input_shape, output_size):
  # input shape is [600, 800, 3]
  model = Sequential()
  # normalization pixel value to [-1, 1]
  model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape,
    output_shape=input_shape))
  # 1 by 1 convolution, potentially it counts for a color space transformation
  # in which feature selection is easier
  model.add(Conv2D(3, (1, 1), padding="valid", strides=(1, 1), name="conv2d0"))
  model.add(ELU())
  # 3@[600, 800]
  model.add(Conv2D(32, (5, 5), padding="valid", strides=(2, 2), name="conv2d1"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 32@[298, 398]
  model.add(Conv2D(64, (5, 5), padding="valid", strides=(2, 2), name="conv2d2"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 64@[147, 197]
  model.add(
    Conv2D(128, (5, 5), padding="valid", strides=(2, 2), name="conv2d3"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 128@[72, 97]
  model.add(
    Conv2D(256, (5, 5), padding="valid", strides=(2, 2), name="conv2d4"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 256@[34, 47]
  model.add(
    Conv2D(512, (5, 5), padding="valid", strides=(2, 2), name="conv2d5"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 512@[15, 22]
  model.add(
    Conv2D(1024, (5, 5), padding="valid", strides=(2, 2), name="conv2d6"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 10240@[6, 9]
  model.add(
    Conv2D(1024, (5, 5), padding="valid", strides=(2, 2), name="conv2d7"))
  model.add(Dropout(0.2))
  model.add(ELU())
  # 10240@[1, 3]
  model.add(Flatten())
  model.add(Dropout(0.2))
  model.add(ELU())

  model.add(Dense(1024, name="dense0"))
  model.add(Dropout(0.5))
  model.add(ELU())
  model.add(Dense(512, name="dense1"))
  model.add(Dropout(0.5))
  model.add(ELU())
  model.add(Dense(output_size, name="dense2"))
  model.add(Activation('softmax'))

  adam = Adam(lr=0.0001)
  model.compile(optimizer=adam, loss="mse")

  return model


if __name__ == "__main__":
  output_folder = "/fig/home/lei/carnd/CarND-Capstone/CarND-Capstone/ros/src/tl_detector/outputs"

  training_dataset_folder = "/data/carnd/CarND-Capstone/dataset/20170922_0_lei"
  validation_dataset_folder = "/data/carnd/CarND-Capstone/dataset/20170916_0_lei"

  RED_TRAFFIC_LIGHT_FOLDER_NAME = "red"
  GREEN_TRAFFIC_LIGHT_FOLDER_NAME = "green"
  YELLOW_TRAFFIC_LIGHT_FOLDER_NAME = "yellow"

  RED_PREDICTION_CLASS_INDEX = 1
  GREEN_PREDICTION_CLASS_INDEX = 0
  YELLOW_PREDICTION_CLASS_INDEX = 2

  training_dataset_size = 0
  for root, dirs, files in os.walk(training_dataset_folder):
    training_dataset_size += len(files)

  training_red_dataset_path = \
    os.path.join(training_dataset_folder, RED_TRAFFIC_LIGHT_FOLDER_NAME)
  training_red_dataset_size = len(
    [name for name in os.listdir(training_red_dataset_path) if
     os.path.isfile(os.path.join(training_red_dataset_path, name))])

  training_green_dataset_path = \
    os.path.join(training_dataset_folder, GREEN_TRAFFIC_LIGHT_FOLDER_NAME)
  training_green_dataset_size = len(
    [name for name in os.listdir(training_green_dataset_path) if
     os.path.isfile(os.path.join(training_green_dataset_path, name))])

  training_yellow_dataset_path = \
    os.path.join(training_dataset_folder, YELLOW_TRAFFIC_LIGHT_FOLDER_NAME)
  training_yellow_dataset_size = len(
    [name for name in os.listdir(training_yellow_dataset_path) if
     os.path.isfile(os.path.join(training_yellow_dataset_path, name))])

  validation_dataset_size = 0
  for root, dirs, files in os.walk(validation_dataset_folder):
    validation_dataset_size += len(files)

  print("training_dataset_size:" + str(training_dataset_size))
  print("training_red_dataset_size:" + str(training_red_dataset_size))
  print("training_green_dataset_size:" + str(training_green_dataset_size))
  print("training_yellow_dataset_size:" + str(training_yellow_dataset_size))
  print("validation_dataset_size:" + str(validation_dataset_size))
  assert (training_dataset_size > 0)
  assert (training_red_dataset_size > 0)
  assert (training_green_dataset_size > 0)
  assert (training_yellow_dataset_size > 0)
  assert ((training_red_dataset_size
           + training_green_dataset_size
           + training_yellow_dataset_size)
          == training_dataset_size)
  assert (validation_dataset_size > 0)

  training_class_weight = \
    {RED_PREDICTION_CLASS_INDEX: 1.0 / training_red_dataset_size,
     GREEN_PREDICTION_CLASS_INDEX: 1.0 / training_green_dataset_size,
     YELLOW_PREDICTION_CLASS_INDEX: 1.0 / training_yellow_dataset_size}
  print("training_class_weight")
  print(training_class_weight)
  assert (training_class_weight[RED_PREDICTION_CLASS_INDEX] > 0)
  assert (training_class_weight[GREEN_PREDICTION_CLASS_INDEX] > 0)
  assert (training_class_weight[YELLOW_PREDICTION_CLASS_INDEX] > 0)

  red_training_percentage = \
    training_class_weight[
      RED_PREDICTION_CLASS_INDEX] * training_red_dataset_size
  green_training_percentage = \
    training_class_weight[
      GREEN_PREDICTION_CLASS_INDEX] * training_green_dataset_size
  yellow_training_percentage = \
    training_class_weight[
      YELLOW_PREDICTION_CLASS_INDEX] * training_yellow_dataset_size
  assert (abs(red_training_percentage - green_training_percentage) < 1e-10)
  assert (abs(red_training_percentage - yellow_training_percentage) < 1e-10)

  input_shape = [600, 800, 3]
  output_size = 3

  BATCH_SIZE = 32
  samples_per_epoch = training_dataset_size
  nb_epoch = 50
  nb_val_samples = validation_dataset_size
  workers = 28

  train_data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=10,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=False)
  train_generator = train_data_gen.flow_from_directory(
    training_dataset_folder,
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    shuffle=True)
  print("train_generator.class_indices")
  print(train_generator.class_indices)
  assert (train_generator.class_indices[RED_TRAFFIC_LIGHT_FOLDER_NAME]
          == RED_PREDICTION_CLASS_INDEX)
  assert (train_generator.class_indices[GREEN_TRAFFIC_LIGHT_FOLDER_NAME]
          == GREEN_PREDICTION_CLASS_INDEX)
  assert (train_generator.class_indices[YELLOW_TRAFFIC_LIGHT_FOLDER_NAME]
          == YELLOW_PREDICTION_CLASS_INDEX)

  validate_data_gen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    fill_mode="nearest",
    horizontal_flip=False,
    vertical_flip=False)
  validate_generator = validate_data_gen.flow_from_directory(
    validation_dataset_folder,
    target_size=input_shape[:2],
    batch_size=BATCH_SIZE,
    shuffle=False)
  print("validate_generator.class_indices")
  print(validate_generator.class_indices)
  assert (validate_generator.class_indices[RED_TRAFFIC_LIGHT_FOLDER_NAME]
          == RED_PREDICTION_CLASS_INDEX)
  assert (validate_generator.class_indices[GREEN_TRAFFIC_LIGHT_FOLDER_NAME]
          == GREEN_PREDICTION_CLASS_INDEX)
  assert (validate_generator.class_indices[YELLOW_TRAFFIC_LIGHT_FOLDER_NAME]
          == YELLOW_PREDICTION_CLASS_INDEX)

  model = get_model(input_shape, output_size)

  model_check_point = ModelCheckpoint(
    os.path.join(output_folder, "model.{epoch:02d}-{val_loss:.6f}.h5"),
    verbose=1)
  callbacks_list = [model_check_point]

  model.fit_generator(
    train_generator,
    class_weight=training_class_weight,
    steps_per_epoch=samples_per_epoch / BATCH_SIZE,
    epochs=nb_epoch,
    validation_data=validate_generator,
    validation_steps=nb_val_samples / BATCH_SIZE,
    workers=workers,
    callbacks=callbacks_list)
  print("Saving model weights and configuration file.")

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  model.save(os.path.join(output_folder, "model.h5"))

  with open(os.path.join(output_folder, "model.json"), "w") as json_file:
    json_file.write(
      simplejson.dumps(simplejson.loads(model.to_json()), indent=4))
