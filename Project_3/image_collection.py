'''
FUNCTION: Use webcam to take picture of opened eyes and closed eyes

DESCRIPTION: OpenCV is used to capture frame of a video feed for open eyes
				and closed eyes.
			 The images are saved to the appropriate directory for the object
			 	classifier to use
		 	 Images are then split into training and validation folders

DIRECTORY:

		CWD--
			|
			|
			-- Closed_Eyes--
			|			   |
			|			   -- *Where Closed Eye images are placed after video feed
			|
			-- Open_Eyes--
			|			 |
			|			 -- *Where Open Eye images are placed after video feed
			|
			-- train --
			|		  |
			|		  -- Closed_Eyes--
			|		  |				 | 
			|		  |				 -- *Where closed eye images are placed after splitting data into training/validation
			|		  |
			|		  -- Open_Eyes--
			|		  			   |
			|		  			   -- *Where open eye images are placed after splitting data into training/validation
			|
			-- validation --
						   |
						   -- Closed_Eyes--
						   |			  |
						   |			  -- *Where closed eye images are placed after splitting data into training/validation
						   |
						   -- Open_Eyes--
						   				|
						   				-- *Where open eye images are placed after splitting data into training/validation
						   				 

'''


import numpy as np 
import os
import cv2
import time
from tqdm import tqdm
from shutil import copyfile
import random

# Save images as JPG
filetype = '.jpg'
# Time in between taking pictures
sleep_interval = 0.1


def make_directory(directory):
	'''Make new directory if it doesn't exist'''
	if not os.path.exists(directory):
		os.mkdir(directory)


def take_pictures(destination_folder, imagetype_folder, start_index=0, num_pictures=10, showRecordings=True):
	'''
	Open a video feed and save each frame to a specified folder
	Arguements:
		destination_folder: directory that will contain "Opened" and "Closed" images
		imagetype_folder: name of folder to place images; "Open" or "Close"
		start_index: integer to start range at; only need to change to have different file names
		num_pictures: number of images to save
		showRecordings: "True" displays the recording in a window; "False" does not show recording
	'''
	# Create directory if it doesn't exist
	make_directory(destination_folder)

	# Open video stream and save images
	video_stream = cv2.VideoCapture(0)
	for i in tqdm(range(start_index, start_index+num_pictures)):
		_, frame = video_stream.read()
		if showRecordings:
		  cv2.imshow('image',frame)
		cv2.imwrite(destination_folder+'/'+imagetype_folder+str(i)+filetype, frame)
		time.sleep(sleep_interval)
		if cv2.waitKey(1) & 0xFF  == ord('q'):
		  break
	del(video_stream)
	for i in range(1, num_pictures):
		cv2.destroyAllWindows()
		cv2.waitKey(1)



# Get current working directory
current_folder = os.getcwd()

# Create the directories needed for Open Eyes images and take pictures
open_eyes_folder = 'Open_Eyes'
destination_folder = os.path.join(current_folder, open_eyes_folder)
take_pictures(destination_folder, open_eyes_folder, start_index=0)

# 2 second pause before taking Eyes Closed pictures
time.sleep(2)

# Create the directories needed for Closed Eyes images and take pictures
closed_eyes_folder = 'Closed_Eyes'
destination_folder = os.path.join(current_folder, closed_eyes_folder)
take_pictures(destination_folder, closed_eyes_folder, start_index=0)



# Create training and validation directories
train_dir = os.path.join(current_folder, 'train')
validation_dir = os.path.join(current_folder, 'validation')

# Create labeled folders within training directory
train_open_dir = os.path.join(train_dir, 'Open_Eyes')
train_closed_dir = os.path.join(train_dir, 'Closed_Eyes')

# Create labeled folders within validation directory
validation_open_dir = os.path.join(validation_dir, 'Open_Eyes')
validation_closed_dir = os.path.join(validation_dir, 'Closed_Eyes')

# Make any directory that doesn't already exist
make_directory(train_dir)
make_directory(validation_dir)
make_directory(train_open_dir)
make_directory(train_closed_dir)
make_directory(validation_open_dir)
make_directory(validation_closed_dir)



def split_data(subset_dir, main_dir=current_folder, validation_ratio=0.2):
	'''
	Go through the Opened and Closed Eye images and split them into training and validation batches
	Arguments:
		subset_dir: types of images you are working with; "Opened" or "Closed"
		main_dir: current working directory
		validation_ratio: fraction of images you want to be validation
	'''
	# Access directory of labeled images
	main_dir = os.path.join(main_dir, subset_dir)
	# Choose random image from directory
	fnames = os.listdir(main_dir)
	train_fnames = np.random.choice(fnames, int((1-validation_ratio)*len(fnames)), replace=False)
	# Choose any image that was not choosen in above line
	validation_fnames = [i for i in fnames if i not in train_fnames]

	# Create directory variables
	destination_train_dir = os.path.join(train_dir, subset_dir)
	destination_validation_dir = os.path.join(validation_dir, subset_dir)

	# Copy files from original location to final location
	for file in train_fnames:
		original = main_dir+'/'+file
		target = train_dir+'/'+subset_dir+'/'+file
		copyfile(original, target)

	for file in validation_fnames:
		original = main_dir+'/'+file
		target = validation_dir+'/'+subset_dir+'/'+file
		copyfile(original, target)


# Split data into training and validation folders 
split_data(subset_dir='Closed_Eyes')
split_data(subset_dir='Open_Eyes')







