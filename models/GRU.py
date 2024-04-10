import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_shape, hidden_size, num_classes):
        super(GRU, self).__init__()
        
        self.gru = nn.GRU(input_size=input_shape[-1], hidden_size=hidden_size,
                          num_layers=2, batch_first=True, bidirectional=False)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # GRU layers
        out, _ = self.gru(x)
        
        # Extract the last output of the second GRU layer
        out = out[:, -1, :]  # Get the output of the last time step
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out
