#####################################

# 3-Block VGG Model using Keras
# Adapted from a tutorial by Jason Brownlee
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
# Jacob Buckelew, Jenny Goldsher, Griffin Lehrer, Maria Morales


#####################################
import sys
# Import necessary modules from the Keras Library
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

def define_model():
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
  # Basically, 20% of neurons will be dropped out from both the forward and backward phase
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
  # Perform stochastic gradient descent in the backpropagation phase
  # Use a learning rate 0.001 which is basically how much the model will be changed
  # in response to error each time weights are updated after a pass through the model
  # Momentum of 0.9 is the speed of backpropagation
   opt = SGD(lr=0.001, momentum=0.9)
  # Compile will configure the model for training by setting its optimizer for stochastic
  # gradient descent which is default_opt
  # 2nd arg is the loss function which is binary_cross entropy
  # This function helps in finding the difference between output probability and the actual value
  # The result of the loss function will be important in determining how the weights should
  # adjust during backpropagation
  # The cross entropy function is binary since we are deciding between 2 classes, cats vs dogs
  # metrics indicates that we are focusing on accuracy of the model as our main metric of interest
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

# plot the learning curves using summarize_diagnostics given the history of a pass through the network
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness():
	# Create a model
	model = define_model()


	# IMAGE GENERATION
	# ---------------------------
	# create two data generators
	# One generator will be for the *Augmented* training data in order to add noise to the images
	# Other image data generator will be for test data and will not be augmented

	# Small width and height chances of 15% and include random horizontal flips
	# Augmentation will help the model focus on individual features independent of position and other nearby pixels
	test_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.15, height_shift_range=0.15, 
		horizontal_flip = True)
	train_datagen = ImageDataGenerator(rescale=1.0/255.0)


	# Generate batches of data for training and test sets
	# Iterate through the images in the dogs-vs-catstrain and dogs-vs-catstest1 directories
	train_it = train_datagen.flow_from_directory('/content/gdrive/My Drive/Kaggle/dogs-vs-cats-for drive/train/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('/content/gdrive/My Drive/Kaggle/dogs-vs-cats-for drive/train/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))


		
	# fit model to the training set
	# first arg is the training set
	# 2nd arg indicates steps per epoch, so the number of entries in train_it
	# 3rd arg for validation_data is test_it since we  want to validate the network
	# using the test data
	# 4th arg indicates validation steps, so number of entries in test_t
	# 5th arg is to Perform 50 epochs, which means 50 iterations of both feeding the data into the network
	# and performing backpropagation
	# 
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=1)


	# evaluate model using the test data and print out accuracies
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	# Create the learning curve
	summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()