# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:48:25 2021

@author:Regen
"""


#Load libraries
import os, torch, torchvision, glob, pathlib, time
import numpy as np
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable


# Check for CUDA compatibility (Dedicated graphics card)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Debug message to specific if running on CPU or a GPU
print(f"[-] Info :: Device used for computations: {device}")

# Add a transformation template to be applied to input images to prepare them for the CNN
transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

#Path for training and testing directory
#! Perhaps use a relative path to the files. I have changed this to work for my system. The folder the "Weather Prediction ML.py" is the same folder as the "Weather predict" folder with the train and test dataset in
trainPath=r"C:\Users\Toshiba\Desktop\Regen\Pyproblems\Weather predict\Train"
testPath=r"C:\Users\Toshiba\Desktop\Regen\Pyproblems\Weather predict\Train"
predictPath=r"C:\Users\Toshiba\Desktop\Regen\Pyproblems\Weather predict\Test"

# Data loader for the training data
trainLoader=DataLoader(
    torchvision.datasets.ImageFolder(trainPath,transform=transformer),
    batch_size=64, shuffle=True
)

# Data loader for the test data
testLoader=DataLoader(
    torchvision.datasets.ImageFolder(testPath,transform=transformer),
    batch_size=32, shuffle=True
)

# Folder to the train directory (within the train directory the classifications are identified by subfolders. For example inside the root path there is a folder called "rain" for the rain classification)
root = pathlib.Path(trainPath)
# Read the subfolders in the train directory and set the name of the folder as a classification
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
# Debug message to output the classifications
print(f"[-] Info :: Classifications: {classes}")

# Class that is a sub-class of pytorch's neural network class, in this usecase this subclass represents a CNN
class ConvNet(nn.Module):

    # Constructor for the CNN
    def __init__(self,num_classes=4):
        # Invoke the super constructor (The super classes constructor (pytorch's nn.Module class) )
        super(ConvNet,self).__init__()

        # LAYER ONE
        # Input:                    An image of shape (256,3,150,150)
        # Output:                   Image after a convolution filter, ((w-f+2P)/s) +1, has been applied
        # Filter size (kernel):     3

        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()
        
        # POOLING - Theory is a little longer then can be bothered to read (Changes sample to smaller regions to allow for greater pattern recognition - or something)
        # Input:                Output of the previous layer
        # Output:               Reduced images size by a factor of 2 (P.S. Following comments not sure but if you say so ;) )
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        # LAYER TWO (or three depending on if pooling counts as a layer, I am assuming it doesn't...)
        # Input:                Output from pooling
        # Output:               (At this point I have no idea, a model would probably not be a bad idea)

        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()

        # LAYER THREE
        # Input:                Values of previous neurons multiplied by the weights to them and then only 0 or positive values
        # Output:               Input with width and height reduced by 2, then increased number of features
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()

        # LAYER FOUR
        # Input:                Input values for the neural network
        # Output:               The input values
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)

    # Function to "feed forward" this moves the data through the neural network from the input layer to the output layer
    def forward(self,input):
        # LAYER ONE
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        # POOLNG
        output=self.pool(output)
            
        # LAYER TWO
        output=self.conv2(output)
        output=self.relu2(output)
            
        # LAYER THREE
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        
        # Output of the neural network in matrix form
        output=output.view(-1,32*75*75)
            
        # Feed inside fully connected layer and find final output
        output=self.fc(output)

        return output

# Call coordinate class with the number of output matrices
model=ConvNet(num_classes=6).to(device)

# Optmiser and loss function
optimiser=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
lossFunction=nn.CrossEntropyLoss()

# Number of training and testing iterations
numEpochs=10

# Get the number of images in both training and testing path
trainCount=len(glob.glob(trainPath+'/**/*.jpg'))
testCount=len(glob.glob(testPath+'/**/*.jpg'))

# Debug message to output the number of images in both datasets
print(f"[-] Info :: Number of training images: {trainCount}, and the number of testing images: {testCount}")

# Initialise a variable to hold the best accuracy
maxAccuracy = 0.0

# Initialise a list to hold the duration of each epoch
epochDurations = []

# Train network by iterating an epochNum amount of times
for epoch in range(numEpochs):

    # Initialise a time to represent the start of the epoch iteration
    startTime = time.time()
    
    # Set the model to training mode (This means that the model keeps the layer's dropout and normalisation)
    model.train()
    
    # Initialise variables for the training accuracy and loss 
    trainAccuracy=0.0
    trainLoss=0.0
    
    # Iterate through the images and the classification (subfolder the image is located)
    for i, (images,labels) in enumerate(trainLoader):
        # Check for cuda availibility
        if torch.cuda.is_available():
            # If availible then use CUDA (as significantly quicker)
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        
        # Set the gradients to zero
        optimiser.zero_grad()
        # Set the output of neural network for set of images
        outputs=model(images)

        # Calculates loss function
        loss=lossFunction(outputs,labels)
        loss.backward()
        optimiser.step()
        
        trainLoss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        trainAccuracy+=int(torch.sum(prediction==labels.data))

    # Calculate the accuracy of the neural network output as a percentage       
    trainAccuracy=trainAccuracy/trainCount
    # Calculate the loss of the neural network output as a percentage       
    trainLoss=trainLoss/trainCount
    
    # Call a function to evaluate the output of the neural network
    model.eval()
    
    # Initialise a variable to hold the accuracy of the test output of the neural network
    testAccuracy=0.0
    # Same as above, iterate through test data and input it to the neural network
    for i, (images,labels) in enumerate(testLoader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())

        # Set the output of the neural network for a given set of images
        outputs = model(images)

        _,prediction=torch.max(outputs.data,1)
        testAccuracy+=int(torch.sum(prediction==labels.data))
        
    # Calculate the test accuracy as a percentage
    testAccuracy=testAccuracy/testCount
    
    # Debug message to output epoch iteration info
    print(f"[-] Info :: Epoch: [{epoch}] Train Loss: [{trainLoss}] Train Accuracy: [{trainAccuracy}] Test Accuracy: [{testAccuracy}]")
    
    # Check if the current test accuracy is greater
    if testAccuracy>maxAccuracy:
        # If greater then same the model, as it is the best currently
        torch.save(model.state_dict(),'best_checkpoint.model')
        maxAccuracy = testAccuracy

    #! Append duration of the iteration (epoch) to the list
    epochDurations.append(time.time() - startTime)
        
        

#! Debugging message to show the number of iterations, the average time for each epoch and the array holding the epoch execution times
print(f"[-] Info :: Number of iterations: {len(epochDurations)}, average time: {np.array(epochDurations).mean()}, times: {np.array(epochDurations)}")

#! For debugging, we save the timings in a .dat file (BINARY WRITE (and append) hence the 'ab' in the open() statement)
with open("Epoch-timings.dat", "ab") as File:
    #! Use numpy's save array to file feature to save the epoch execution time array as a single line
    np.savetxt(File, np.array(epochDurations), newline=',')
    #! Add a newline character to the file as next time file should be written to it should be on a new line
    File.write(b"\n")

