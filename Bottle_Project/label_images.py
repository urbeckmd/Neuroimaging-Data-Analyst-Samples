'''
FUNCTION: Move images of bottles to appropriately labeled directory

DESCRIPTION: All images start out in a single Folder.
             Iterate through images and use keys to move them to appropriate
                folder for the model to use.

NOTES: Labels are:
            "FOAM" - foam is present in the bottle. Cannot get accurate volumetric reading
            "GREATER" - volume is greater than 20 mL
            "LESS" - volume is less than 20 mL
'''
import os
import cv2
import shutil



def directory(path):
    '''
    Check if labeled directories already exist. If not, create them.
    Arguments:
        path: os path
    '''
    if not os.path.isdir(path):
        os.mkdir(path)


def move_image(image, label):
    '''
    Move image from original folder to labeled folder
    Arguments:
        image - image file
        label - name of labeled folder
    '''
    shutil.move(f"images/{image}", f"data/{label}/{image}")


# Set directory that contains label Folders
data_dir = os.path.join(os.getcwd(), 'data')
# Set Folder to hold "Foam" images
foam_dir = os.path.join(data_dir, 'Foam')
# Set Folder to holder "Greater Than" images
greater_dir = os.path.join(data_dir, 'Greater')
# Set Folder to holder "Less Than" images
less_dir = os.path.join(data_dir, 'Less')
# If an of the above directories do not exist, create them
directory(data_dir)
directory(foam_dir)
directory(greater_dir)
directory(less_dir)

# Label all the images by moving them to appropriate folder
for image in os.listdir("images"):
    # load it with openCV and resize the image
    result = cv2.imread(f'images/{image}')
    width = int(result.shape[1]*.2)
    height = int(result.shape[0]*.2)
    result = cv2.resize(result, (width, height))

    # Display the image
    cv2.imshow('Space-FOAM  Backspace-GREATER Tab-LESS', result)
    
    # Move image to label folder depending on which key is pressed
    k = cv2.waitKey(0)
    if k == 32: # Space - FOAM
        move_image(image, "Foam")
    elif k == 8: # Backspace - GREATER
        move_image(image, "Greater")
    elif k == 9: # Tab - LESS
        move_image(image, "Less")
    elif k == 27: # Escape - Break the loop
        break
    cv2.destroyWindow('image')
