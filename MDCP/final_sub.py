################################################

# final_sub.py organizes the subdirectories of data for the final model
# Jacob Buckelew, Jenny Goldsher, Griffin Lehrer, Maria Morales

################################################
# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
# create directories
dataset_home = 'finalize_dogs_vs_cats_for_drive/'
# create label subdirectories
#labeldirs = ['dogs/', 'cats/']
#for labldir in labeldirs:
	#newdir = dataset_home + labldir
	#makedirs(newdir, exist_ok=True)
# copy training dataset images into subdirectories
src_directory = '/Users/mariamorales/Documents/Capstome_Projext/dogs-vs-cats-for-drive/train/train/dogs'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	if file.startswith('cat'):
		dst = dataset_home + 'cats/'  + file
		copyfile(src, dst)
	elif file.startswith('dog'):
		dst = dataset_home + 'dogs/'  + file
		copyfile(src, dst)