from keras.models import Model
import sys
from tensorflow import keras
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.optimizers import Adam
def define_final_model():
  # The CNN model is built using Sequential() which will link a series of layers together
  model = Sequential()
  # 3 Block Model consists of 3 Blocks each containing a Conv2D and MaxPooling2D layer
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
  model.add(MaxPooling2D((2, 2)))
  # Dropout occurs at the end of each layer except the output layer to prevent overfitting
  # Basically, 10% of neurons will be dropped out from both the forward and backward phase
  # This prevents neurons from learning based on context and their position around other neurons
  model.add(Dropout(.20))
  
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  
  model.add(Dropout(.30))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  
  model.add(Dropout(.40))
  model.add(Flatten())

  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dropout(.50))
  
  model.add(Dense(1, activation='sigmoid'))
  # maria = 2
  # compile model
  # opt = SGD(lr=0.001, momentum=0.9)
  default_opt = Adam(learning_rate=0.001)
  model.compile(optimizer=default_opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

  # run the test harness for evaluating a model
def run_final_test_harness():
# define model
  model = define_final_model()
  # create data generator
  datagen = ImageDataGenerator(featurewise_center=True)
  # specify imagenet mean values for centering
  datagen.mean = [123.68, 116.779, 103.939]
  # prepare iterator
  train_it = datagen.flow_from_directory('/content/gdrive/MyDrive/Kaggle/finalize_dogs_vs_cats_for_drive/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
  # fit model
  model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=1)
  # save model
  model.save('final_model.h5')

  run_final_test_harness()