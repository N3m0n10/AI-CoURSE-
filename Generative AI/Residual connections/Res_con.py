import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module): #@save
    """'The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()

        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
        stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
            stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):

        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)
            Y += X

        return F.relu(Y)
    

class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ResidualMLP, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        identity = x # Store original input for the residual connection

        # First fully connected layer
        out = F.relu(self.fc1(x))
        # Second fully connected layer (with residual connection)
        out = F.relu(self.fc2(out) + identity) # Adding the original input (residual connection)
        # Final layer
        out = self.fc3(out)
        
        return out
    
#TODO: add this, and also VGG, etc..., to a utils.py in Generative AI folder