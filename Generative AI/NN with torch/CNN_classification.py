"Provided by professor"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Data transformation (convert to tensor and normalize)
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]) # black and white

# Download and load training dataset
train_data = datasets.MNIST(root='./data', train=True,
    transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
    shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolution layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                                kernel_size=3, stride=1, padding=1)
        # Convolution layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                                kernel_size=3, stride=1, padding=1)
        # Fully connected layers
        # 7x7 is the size of the image after pooling layers
        self.fc1 = nn.Linear(7*7*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1: Convolution -> Activation -> Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) # 2x2 max pooling
        # Layer 2: Convolution -> Activation -> Pooling

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) # 2x2 max pooling

        #print(x.shape)

        # Flatten the tensor
        x = x.view(-1, 7*7*64)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

model = SimpleCNN()

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