'''
 Image segmentation model

 FUNCTION: Identify and outline a specific object in an image

 DESCRIPTION: Pytorch Image Segmentation model was used to locate
                boat ramps inside images
              The project was built to identify boat ramps, but the model 
                can be trained for any object
'''

import os
import numpy as np 
import torch 
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T 
import utils
from engine import train_one_epoch, evaluate
import warnings
warnings.filterwarnings('ignore')

# List folders where images/masks are located
img_folder = 'RampImages'
mask_folder = 'RampMasks'
dataset_folder = 'RampDataset'

class BoatRampDataset(object):
    def __init__(self, root, transforms):
        '''Initialize object'''
        self.root = root
        self.transforms = transforms
        # Load image files and sort them so images and masks are aligned together
        self.imgs = list(sorted(os.listdir(os.path.join(root, img_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(root, mask_folder))))

    def __getitem__(self, idx):
        '''Code to call when object is iterated through'''
        img_path = os.path.join(self.root, img_folder, self.imgs[idx])
        mask_path = os.path.join(self.root, mask_folder, self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # Find number of objects inside image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            '''find corners of bounding box'''
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    '''Load a pretrained model and modify the structure to fit our needs'''
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get number of input features to feed into classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace head of model with new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get number of input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layers = 256
    # Replace mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)

    return model

def get_transform(train):
    '''Define transformations to apply'''
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return  T.Compose(transforms)

# Initialize batch size and number of epochs for training
batch_size = 2
num_epochs = 3

# Create the training and testing datasets
dataset = BoatRampDataset(dataset_folder, get_transform(train=True))
dataset_test = BoatRampDataset(dataset_folder, get_transform(train=False))

# Split dataset into training and testing subsets
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-15])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-15:])

# Create Data Loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                          collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0,
                                               collate_fn=utils.collate_fn)


device = torch.device('cpu')
# Set number of classes: Background and Ramp
num_classes = 2
# Create the model
model = get_model_instance_segmentation(num_classes)
model.to(device)
# Create optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# Create dynamic learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


for epoch in range(num_epochs):
    # train on single epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # Update learning rate
    lr_scheduler.step()
    # Evaluate performance on test data 
    evaluate(model, data_loader_test, device=device)

# Save model
torch.save(model, f'{dataset_folder}_rcnn.pt')

# Shows its prediction by displaying image
img, _ = dataset_test[1]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

x = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
x.show()
y = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
y.show()
