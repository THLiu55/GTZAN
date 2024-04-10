import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MLP, self).__init__()
        
        # Calculate the input size based on the input_shape
        input_size = input_shape[1] * input_shape[2]
        
        # Define the layers of the model
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)
        
        # Fully connected layer 1 - 512 units
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Fully connected layer 2 - 256 units
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Fully connected layer 3 - 64 units
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        x = self.softmax(x)
        
        return x
