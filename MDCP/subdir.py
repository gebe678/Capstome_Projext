##########################################################

# Subdir.py Organizes the subdirectories of training and test data located in the MDCP directory
# Jacob Buckelew, Jenny Goldsher, Griffin Lehrer, Maria Morales

##########################################################
import os
import random
# importing shutil module 
import shutil as shu
# create directories
dataset_home = '/Users/mariamorales/Documents/MDCP/dogs-vs-cats'
subdirs = ['train/', 'test1/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['dogs/', 'cats/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		os.makedirs(newdir, exist_ok=True)




# seed random number generator
random.seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.25
# copy training dataset images into subdirectories
src_directory = '/Users/mariamorales/Documents/MDCP/dogs-vs-cats/train'
for file in os.listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random.random() < val_ratio:
		dst_dir = 'test1/'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats/'  + file
		shu.copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs/'  + file
		shu.copyfile(src, dst)