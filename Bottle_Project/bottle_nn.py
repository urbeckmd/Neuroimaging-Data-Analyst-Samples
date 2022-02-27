'''
Image classifier 

Used Pytorch documentation to modify a pretrained CNN to determine the volume range
of blood within a vial

Images were labeled as:
    "Foam" - foam is present within the bottle.
    "Greater" - volume is greater than 20 mL.
    "Less" - volume is less than 20mL.

CNN was trained on 150 images so volume range had to be rather large
More training images would allow for exact mL measures or 5 mL ranges.
'''


import torch 
import torch.optim as optim
import torch.nn.functional as F 
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import json



# Load images into dataset and apply transformations to images
dataset = datasets.ImageFolder(
    'data',
        transforms.Compose([
        transforms.ColorJitter(0.1,0.1,0.1,0.1),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
)


# Create dict that pairs label to numberic value; {foam:0, greater:1, less:2}
class_to_idx = dataset.class_to_idx
# Create dict that pairs numeric value to label {0:foam, 1:greater, 2:less}
ind_to_class = {val: key for key, val in class_to_idx.items()}

# Create JSON file that holds label to numeric value pair
with open('class_to_idx.json','w') as f:
    json.dump(ind_to_class, f, indent=4)



# Split dataset into testing and training
length_testing_dataset = 25
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-length_testing_dataset, length_testing_dataset])



# Turn training dataset into iterable
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# Turn testing dataset into iterable
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True
)



# Load pretrained model
model = models.alexnet(pretrained=True)
# Modify classifier to have 3 output
num_outputs = 3
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_outputs)



# Number of epoch to train model
num_epochs = 10
# name of file to save model to
best_model_path = 'bottle_best_model.pt'
# Variable to store accuracy of best model
best_acc = 0.0

# Use stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



# Run epochs
for epoch in range(num_epochs):
    print(f'Beginning Epoch {epoch+1}')
    for images, labels in iter(train_loader):
        # zero out gradients
        optimizer.zero_grad()
        # run images through model
        outputs = model(images)
        # calculate loss
        loss = F.cross_entropy(outputs, labels)
        # calculate gradient
        loss.backward()
        # update model parameters
        optimizer.step()

    # test model on testing data and calculate an error
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))

    # Calculate an accurancy and save the parameters if accuracy is improved
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f' % (epoch+1, test_accuracy))
    if test_accuracy > best_acc:
        torch.save(model.state_dict(), best_model_path)
        best_acc = test_accuracy
    print(f'Epoch {epoch+1} Complete')
    break
