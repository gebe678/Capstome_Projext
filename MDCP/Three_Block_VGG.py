# baseline model for the dogs vs cats dataset
import sys
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

# define cnn model
def define_model():

	# The CNN model is built using Sequential() which will link a series of layers together
	model = Sequential()

	# 3 Block Model consists of 3 Blocks each containing a Conv2D and MaxPooling2D layer

	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	# Dropout occurs at the end of each layer except the output layer to prevent overfitting
	# Basically, 10% of neurons will be dropped out from both the forward and backward phase
	# This prevents neurons from learning based on context and their position around other neurons
	model.add(Dropout(.10))
  
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
  
	model.add(Dropout(.10))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
  
	model.add(Dropout(.10))
	model.add(Flatten())
  
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(.50))

	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
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
	# define model
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
	train_it = train_datagen.flow_from_directory('/Users/mariamorales/Documents/Capstome_Projext/MDCP/dogs-vs-catstrain/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = test_datagen.flow_from_directory('/Users/mariamorales/Documents/Capstome_Projext/MDCP/dogs-vs-catstest1/',
		class_mode='binary', batch_size=64, target_size=(200, 200))


		
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=0)


	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()