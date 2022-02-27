PROJECT DESCRIPTION: This project uses a Pytorch Image Segmentation model to identify and locate boat ramps in an image.

FILES: Two files are present in this project: image_labeling.py , image_segmentation_model.py

IMAGE_LABELING.PY: This script labels and organizes the raw images so that the model can be trained on the processed images. "LabelMe" (an MIT Software) was used as the labeling tool.

IMAGE_SEGMENTATION_MODEL.PY: This script loads and trains a pre-trained Pytorch segmentation model to locate a boat ramp within an image. 
