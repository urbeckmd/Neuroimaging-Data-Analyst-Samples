'''
Object Classifier

FUNCTION: Predict if eyes are open or closed in an image

DESCRIPTION: Keras was used to create a CNN from scratch.
			 After the model is trained, plots of different metrics
			 	are created to display the models performance.
'''

import numpy as np 
import os
import tensorflow as tf 
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 

# Create variables for directory to image data
current_folder = os.getcwd()
train_dir = os.path.join(current_folder, 'train')
validation_dir = os.path.join(current_folder, 'validation')


# Add specs of model
img_width, img_height = 256, 256
batch_size = 16
epochs = 25
lr = 1e-6

# Create structure of Model
model = models.Sequential()
model.add(layers.Conv2D(128, (3,3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))



# Configure model for training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])



# Create data generators and resize images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# Fill generator with data from directory
train_generator = train_datagen.flow_from_directory(
              train_dir,
              target_size=(img_height, img_width),
              class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
              validation_dir,
              target_size=(img_height, img_width),
              class_mode='categorical')


# Train the model
history = model.fit_generator(
      train_generator,
      epochs=epochs,
      validation_data=validation_generator)

# Save model
model.save('model.h5')

# Access different metrics from training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

# Plot Accuracy over epochs
plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and Validation Acc')
plt.legend()
plt.figure()

# Plot Loss over epochs
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()




















