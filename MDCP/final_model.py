#####################################

# Final Model using Keras
# Adapted from a tutorial by Jason Brownlee
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# Jacob Buckelew, Jenny Goldsher, Griffin Lehrer, Maria Morales


#####################################
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
  # Conv2D is the first type of layer in a block
  # this layer will perform the process of convolution on the input data

  # First arg is the number of filters present in the layer
  # 2nd arg is a 2-tuple for the size of a filter
  # 3rd arg is an activation function that performs after convolution
  # For our case, we use the ReLu activation function which is the rectified linear function
  # 4th arg is an kernel initializer that sets all the values in the kernel weights matrix
  # For our case, we use He initialization which is optimal for ReLu activation functions
  # 5th arg indicates the type of padding used. Same padding means that the output volume remains
  # the same size as the input volume after convolution happens
  # The first Conv2d call will also have an input_shape arg, for the volume of the input, so in this case
  # a 200 x 200 matrix with 3 sheets so 200x200x3
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
  # MaxPooling layer reduces the dimensionality of a matrix but still retains important information
  # Only arg is a pool size which indicates the window size over which to take a maximum value
  # when performing the pooling process

  model.add(MaxPooling2D((2, 2)))
  # Dropout occurs at the end of each layer except the output layer to prevent overfitting
  # Basically, 10% of neurons will be dropped out from both the forward and backward phase
  # This prevents neurons from learning based on context and their position around other neurons
  model.add(Dropout(.20))

  # Each subsequent Conv2d will double in number of filters used to perform convolution
  # Keep all of the same initialization, activation, and padding parameters
  
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  # Gradually increase dropout as data passes through each block to optimize dropout regularization
  model.add(Dropout(.30))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  
  model.add(Dropout(.40))
  # Flatten converts the pool matrix to a single column that is passed to the normal dense layers
  # which don't perform convolution
  model.add(Flatten())

  # Dense layer is a typical Neural Network layer consisting of neurons.
  # First arg is number of neurons which is 128
  # Activation and kernel_initializer remain consistent from before

  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dropout(.50))
  # Last Dense layer will perform the sigmoid function with a single neuron and then provide
  # an output accuracy value
  model.add(Dense(1, activation='sigmoid'))
  # compile model
  # We now use the Adam optimizer which is a stochastic gradient descent method
  # that performs well when given large amounts of data
  default_opt = Adam(learning_rate=0.001)
  # Compile will configure the model for training by setting its optimizer for stochastic
  # gradient descent which is default_opt
  # 2nd arg is the loss function which is binary_cross entropy
  # This function helps in finding the difference between output probability and the actual value
  # The result of the loss function will be important in determining how the weights should
  # adjust during backpropagation
  # The cross entropy function is binary since we are deciding between 2 classes, cats vs dogs
  # metrics indicates that we are focusing on accuracy of the model as our main metric of interest
  model.compile(optimizer=default_opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

  # run the test harness for evaluating a model
def run_final_test_harness():
# define final model
  model = define_final_model()
  # create data generator
  datagen = ImageDataGenerator(featurewise_center=True)
  # specify imagenet mean values for centering
  datagen.mean = [123.68, 116.779, 103.939]
  # prepare iterator 
  # generate batches of data for training data
  train_it = datagen.flow_from_directory('/content/gdrive/MyDrive/Kaggle/finalize_dogs_vs_cats_for_drive/',
    class_mode='binary', batch_size=64, target_size=(200, 200))
  # fit model
  # First arg is the training set
  # Second arg is steps per epoch, so the number of entries in training set
  # Third arg is number of epochs, which means 10 iterations of the feedforward and backpropagation phases
  # Verbose is set to 1 in order to see program run each epoch in real time
  model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=10, verbose=1)
  # save model
  model.save('final_model.h5')



# run test harness
run_final_test_harness()