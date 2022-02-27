PROJECT DESCRIPTION: A Pytorch object classifier was trained to identify the volume of blood inside a vial.

FILES: Two files are present inside this project: label_images.py , bottle_nn.py

LABEL_IMAGES.PY: This script iterates though all the unlabeled images and moves them to the appropriate folder by using specific keys on the keyboard.

BOTTLE_NN.PY: This script loads a pretrained model and trains the model on images of vials of blood. There are three output of the model: Foam is present in bottle, Volume is greater than 20 mL, Volume is less than 20 mL.

