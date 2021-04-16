
# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = '/Users/mariamorales/Documents/Capstome_Projext/dogs-vs-cats/train/'

# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = '/Users/mariamorales/Documents/Capstome_Projext/dogs-vs-cats/train'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	#dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	else:
		dst_dir = 'train/'

	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		copyfile(src, dst)