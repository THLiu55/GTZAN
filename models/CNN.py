import torch.nn as nn
import torch.optim as optim

# Define the PyTorch model class
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNN, self).__init__()
        
        # Flatten the input
        self.flatten = nn.Flatten()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(input_shape[1] * input_shape[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        # Define activation function
        self.relu = nn.ReLU()
        
        # Define softmax activation for the output layer
        self.softmax = nn.Softmax(dim=1)  # dim=1 applies softmax along the columns (classes)
    
    def forward(self, x):
        # Forward pass through the network
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

