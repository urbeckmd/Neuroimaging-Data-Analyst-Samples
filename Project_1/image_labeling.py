'''
FUNCTION: Label images with polygons for image segmentation model

DESCRIPTION: Iterate through raw images to transform them, apply polygons around object, 
             label them, and move them to appropriate folder

NOTES: "LabelMe" annotation tool from MIT was used to add polygons around objects
       Script is built in a way that allows you to stop the program in the middle of labeling
            images then rerun the program to pick up where you left off

DIRECTORY STRUCTURE:

    CWD--
        |
        -- RampDataset--
        |               |
        |               -- *folders will be creates to store images and masks*
        |
        |
        -- temp_folder--
                        |
                        -- *raw images to label*
'''
import os
import cv2
import shutil
from PIL import Image
import sys
from simple_image_download import simple_image_download as simp



# Create folder names to store all the iamges
ROOT_DIR = os.getcwd()
RAW_DIR = os.path.join(ROOT_DIR, 'temp_folder')       # Folder where all the raw, unlabeled images initially are stored
DATASET_NAME = 'RampDataset'                          # Folder that has annotation, images, and mask folders
IMAGE_FOLDER_NAME = 'RampImages'                      # Folder that will store the post-processed images
MASK_FOLDER_NAME = 'RampMasks'                        # Folder that will store the masks of images

# Create Folder to Store Labeled Image
IMAGES_DIR = os.path.join(ROOT_DIR, f'{DATASET_NAME}/{IMAGE_FOLDER_NAME}')
if not os.path.isdir(IMAGES_DIR):
    os.mkdir(IMAGES_DIR)
# Create Folder to Store Masks
MASKS_DIR = os.path.join(ROOT_DIR, f'{DATASET_NAME}/{MASK_FOLDER_NAME}')
if not os.path.isdir(MASKS_DIR):
    os.mkdir(MASKS_DIR)
# Create Folder to Store Annotations
ANNON_DIR = os.path.join(ROOT_DIR, f'{DATASET_NAME}/Annotations')
if not os.path.isdir(ANNON_DIR):
    os.mkdir(ANNON_DIR)




# Get num of images already labeled; used for naming the images
# images have name "ramp_{number}.png", where "{number}" is the number of images already labeled
NUM_IMG_DATASET = len(os.listdir(IMAGES_DIR))

# Get number of images that have not been labeled yet; used to determine if all images have been labeled
NUM_IMG_STORE = 0
for i in os.listdir(RAW_DIR):
    # Ignore .json files. Only want .jpg or .png files
    if '.json' not in i:
        NUM_IMG_STORE += 1





# Transform the images that have not been labeled yet
counter = NUM_IMG_DATASET
for img in os.listdir(RAW_DIR):
    if not str(img.endswith(".json")):
        # Convert to grayscale and resize
        image = cv2.imread(os.path.join(RAW_DIR, img))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = image
        h = gray.shape[0]
        w = gray.shape[1]
        if w>h:
            scale = round(512/w,2) 
        else:
            scale = round(512/h,2)
        width = int(image.shape[1]*scale)
        height = int(image.shape[0]*scale)
        dsize = (width, height) 
        gray = cv2.resize(gray, dsize)

        # Zero out pixels in left and right third of image because object will always be in middle of image
        mask_width = round(width/3)
        halfway = round(width/2)
        half_mask = round(mask_width/2)

        gray[:, 0:halfway-half_mask] = 0
        gray[:, halfway+half_mask:] = 0

        # Save processed image and delete raw image
        cv2.imwrite(os.path.join(RAW_DIR, f'ramp_{counter}.png'),gray)
        os.remove(os.path.join(RAW_DIR, img))
        counter += 1






# Annotate Any Images that have yet to be labeled; Produces a json file in same dir as image
for files in os.listdir(RAW_DIR):
    # Only pay attention to image files
    if files.endswith('.json') == False and '.' in files:
        root_name = files.split('.')[0]
        # Only label images that have not already been labeled
        if f'{root_name}.json' not in os.listdir(RAW_DIR):
            # Launch "LabelMe" on image
            os.system("labelme "+os.path.join(RAW_DIR, files))





# Create Data folder for each labeled image that contains image, mask, and label name files
# Check if all images have been annotated
if len(os.listdir(RAW_DIR)) == NUM_IMG_STORE*2:
    # Use LabelMe to convert json file to Dataset
    for files in os.listdir(RAW_DIR):
        if files.endswith('.json'):
            os.system("labelme_json_to_dataset "+os.path.join(RAW_DIR,files))





# Create Added Object List; Don't use for anything, but necessary for LabelMe library
with open(f'{os.path.join(ROOT_DIR, DATASET_NAME)}/added-object-list.txt','a') as added_file:
    added_file.write('# image name (\\t) object index\n')
    added_file.close()





# Process image and mask from each Dataset and move to final destination
for json_folder in os.listdir(RAW_DIR):
    # Access dataset folders only
    if os.path.isdir(os.path.join(RAW_DIR, json_folder)):
        root_name = json_folder.split('_json')[0]
        # load image and mask into opencv
        for json_contents in os.listdir(os.path.join(RAW_DIR, json_folder)):
            if json_contents == 'label.png':
                mask_img = cv2.imread(os.path.join(RAW_DIR, json_folder, json_contents))
            if json_contents == 'img.png':
                img = cv2.imread(os.path.join(RAW_DIR, json_folder, json_contents))
        height = img.shape[0]
        width = img.shape[1]
        channel = img.shape[2]

        # Turn the mask into an image of values 0 for background and 1 for object
        # This is how the neural network wants the masks structured
        gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_mask, 10, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_objects = len(contours)
        obj_list = []
        for cnt in range(num_objects):
            mask_cnt = cv2.drawContours(mask_img, contours, cnt, (cnt+1, cnt+1, cnt+1), -1)
            x,y,w,h = cv2.boundingRect(contours[cnt])
            obj_list.append([x,y,w,h])
            root_img_name = root_name.split('_')[0]
            num = root_name.split('_')[1]
            # Write processed mask to RampMasks folder
            cv2.imwrite(os.path.join(MASKS_DIR, f'{root_img_name}_mask_{num}.png'), mask_cnt)
        
        obj_gt_lst = '{'+' '+"Ramp "*num_objects+'}'

        # Write data to annotation file; necessary for LabelMe software
        with open(f'{ANNON_DIR}/{root_name}.txt','a') as f:
            f.write('# Compatible with PASCAL Annotation Version 1.00\n')
            f.write(f'Image filename : "{DATASET_NAME}/{IMAGE_FOLDER_NAME}/{root_name}.png"\n')
            f.write(f'Image size (X x Y x C) : {width} x {height} x {channel}\n')
            f.write(f'Database : "The Ramp Database"\n')
            f.write(f'Objects with ground truth : {num_objects} {obj_gt_lst}\n')
            f.write('# Note there may be some objects not included in the ground truth list for they are severe-occluded\n')
            f.write('# or have very small size.')
            f.write('# Top left pixel co-ordinates : (1, 1)\n')
            for i in range(num_objects):
                f.write(f'# Details for Ramp {i+1} ("Ramp")\n')
                f.write(f'Original label for object {i+1} "Ramp" : "{DATASET_NAME}"\n')
                f.write(f'Bounding box for object {i+1} "Ramp" (Xmin, Ymin) - (Xmax, Ymax) : ({obj_list[i][0]}, {obj_list[i][1]}) - ({obj_list[i][0]+obj_list[i][2]}, {obj_list[i][1]+obj_list[i][3]})\n')
                f.write(f'Pixel mask for object {i+1} "Ramp" : "{DATASET_NAME}/{MASK_FOLDER_NAME}/{root_name}_mask.png"\n')
                f.write('\n')
        
        # Move image from Dataset folder to RampImages folder 
        shutil.copy(os.path.join(RAW_DIR, json_folder, 'img.png'), os.path.join(IMAGES_DIR, f'{root_name}.png'))
        
        # Write data to added-object-list; necessary for LabelMe software
        with open(f'{os.path.join(ROOT_DIR, DATASET_NAME)}/added-object-list.txt','a') as added_file_2:
            added_file_2.write(f'{root_name}.png\t{num_objects}\n')




# Convert image type of masks; Otherwise neural network doesnt work
for i in os.listdir(MASKS_DIR):
    img = Image.open(os.path.join(MASKS_DIR, i))
    img = img.convert("L")
    img = img.save(os.path.join(MASKS_DIR, i))
