# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 18:04:13 2021

@author: Regen
"""
# Load libraries 
import torch, os, glob, pathlib, time
import torch.functional as F
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms
from torchvision.models import squeezenet1_1
from torch.autograd import Variable
from PIL import Image
from io import open

# Path to training data
trainPath=r"C:\Users\Toshiba\Desktop\Regen\Pyproblems\Weather predict\Train"
# Set the training data path as a root directory
root = pathlib.Path(trainPath)
# For each classification in the training data folder, add it to a list
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

# Define a CNN class which is a subclass of pytorch's nn.Module
class ConvNet(nn.Module):
    def __init__(self,num_classes=4):
        super(ConvNet,self).__init__()

        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)

        output=output.view(-1,32*75*75)

        output=self.fc(output)
        return output


# Set a transform template to be applied to images
transformer=transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                        [0.5,0.5,0.5])
])

# Load best model
bestModel=torch.load('best_checkpoint.model')
# Instantiate a CNN object
model=ConvNet(num_classes=6)
# Set the new CNN object to have the values of the best model
model.load_state_dict(bestModel)
# Evaulate the CNN
model.eval()

# Prediction function
def predict(img_path, transformer):
    # Load the image
    image = Image.open(img_path)
    # Put the image through transform template
    image_tensor = transformer(image).float()
    # Don't really know, but assume its got to do with the tensor structure, presumably transforming data into the tensor data structure
    image_tensor = image_tensor.unsqueeze_(0)

    # Check if Cuda is available
    if torch.cuda.is_available():
        # If it is then use as quicker then CPU
        image_tensor.cuda()

    # Set the input for the neural network
    input = Variable(image_tensor)
    # Set the output of the CNN
    output = model(input)
    # Intialise a variable to hold the index of classification in output
    index = output.data.numpy().argmax()
    # Set the variable prediction to a classification
    prediction = classes[index]

    # Return the predicted classification
    return prediction

# Set the path to the images to predict for
imagesPath=glob.glob(r"C:\Users\Toshiba\Desktop\Regen\Pyproblems\Weather predict\Test\Rain"+'/*.jpg')

# Debug message for how many images to predict
print(f"[-] Info :: Number of images to predict {len(imagesPath)}")

# Dictionary containing the image file name and its predicted classification
pred_dict = {}

# Iterate through images
for i in imagesPath:
    # For each image predict what its classification is
    pred_dict[i[i.rfind('/')+1:]] = predict(i, transformer)

# Output the predictions
print(pred_dict)

# OPTIONAL (Uncomment next line for pause, useful if just wanting to execut on idle)
#time.sleep(100)


