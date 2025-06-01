"Provided by professor"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]
"""

"""
from torchvision.datasets import ImageFolder

dataset = ImageFolder(root='path/to/images', transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
"""

# Data transformation (convert to tensor and normalize)
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]) # black and white

# Download and load training dataset
train_data = datasets.MNIST(root='./data', train=True,
    transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
    shuffle=True)
test_data = datasets.MNIST(root='./data',train=False,
    download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
    shuffle=True)

testloader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,  # No need to shuffle for evaluation
    num_workers=2
)

class DropoutCNN(nn.Module):
    def __init__(self):
        super(DropoutCNN, self).__init__()

        # Convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=3, stride=1, padding=1)
        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=3, stride=1, padding=1)
        
        ## Fully connected layers using LazyLinear
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.LazyLinear(10)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)  # dropout probability
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):

        # Layer 1: Convolution -> Activation -> Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # 2x2 max pooling

        # Layer 2: Convolution -> Activation -> Pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # 2x2 max pooling

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) # Apply dropout after the first FC layer
        x = self.fc2(x)
        x = self.dropout2(x) # Apply dropout after the second FC layer

        return x
    
model = DropoutCNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, labels in train_loader:

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    model.eval() # Set the model to evaluation mode
    with torch.no_grad(): # disable gradients computation
        inputs, labels = next(iter(testloader))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total
        avg_loss = loss.item()
        model.train()
        print(f"Acuracy: {accuracy:.2f}%")
        print("Average loss: ", avg_loss)