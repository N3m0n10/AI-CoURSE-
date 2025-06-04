"Gemini version"
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Data transformation (convert to tensor and normalize)
# This prepares the MNIST images for input into the neural network.
# transforms.ToTensor() converts PIL Image or numpy.ndarray to a PyTorch FloatTensor.
# transforms.Normalize((0.5,), (0.5,)) normalizes the pixel values to a range of [-1, 1].
# For grayscale images like MNIST, we use single values for mean and standard deviation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalizes pixel values for black and white images
])

# Download and load training dataset
# datasets.MNIST downloads the MNIST dataset if not already present.
# train=True specifies the training split.
# transform applies the defined transformations to the images.
# torch.utils.data.DataLoader creates an iterable over the dataset,
# allowing for batch processing and shuffling during training.
train_data = datasets.MNIST(root='./data', train=True,
    transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
    shuffle=True)

# Modified vgg_block function
# This function is designed to create a single VGG-like convolutional block.
# A standard VGG block consists of a sequence of convolutional layers followed by a ReLU activation,
# and then a single max-pooling layer to reduce spatial dimensions.
#
# Original `vgg_block` had `MaxPool2d` inside the loop and `nn.LazyLinear` at the end.
# - `MaxPool2d` inside the loop would lead to very rapid downsampling, making the input
#   too small for subsequent convolutions if `num_convs` is large.
# - `nn.LazyLinear` at the end means the block would output a 1D tensor, preventing
#   it from being stacked as a feature extractor with other convolutional blocks.
#
# To create a functional VGG-like architecture, these adjustments were necessary.
# This modification ensures `vgg_block` acts as a proper feature extraction unit.
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        # nn.LazyConv2d is used to automatically infer the number of input channels
        # on the first forward pass. This is convenient when the input shape
        # might vary or when building a flexible architecture.
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    # A single MaxPool2d is applied after all convolutional layers within this block.
    # This reduces the spatial dimensions (height and width) of the feature maps.
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, num_classes=10, num_vgg_blocks=2, convs_per_block=[1, 1], channels_per_block=[32, 64], num_fc_layers=2):
        """
        Initializes the VGG-like neural network.

        Args:
            num_classes (int): The number of output classes (e.g., 10 for MNIST digits).
            num_vgg_blocks (int): The number of VGG-like convolutional blocks ('n' in "n VGG").
            convs_per_block (list): A list where each element specifies the number of
                                    convolutional layers within each VGG block.
            channels_per_block (list): A list where each element specifies the output
                                       channels for the convolutional layers in each VGG block.
            num_fc_layers (int): The number of fully connected layers in the classifier ('m' in "m FC").
        """
        super(VGG, self).__init__()

        # Feature extraction layers (n VGG blocks)
        # This sequential module builds the convolutional part of the network.
        # It stacks multiple `vgg_block` instances, each designed to extract features
        # at different levels of abstraction.
        features_layers = []
        for i in range(num_vgg_blocks):
            # Each vgg_block is configured with the specified number of convolutional layers
            # and output channels for that block.
            features_layers.append(vgg_block(convs_per_block[i], channels_per_block[i]))
        self.features = nn.Sequential(*features_layers)

        # Classifier layers (m FC layers)
        # This sequential module builds the fully connected layers for classification.
        # It takes the flattened output from the feature extractor and maps it to
        # the final class probabilities.
        classifier_layers = []
        # The first fully connected layer uses nn.LazyLinear to infer its input size,
        # as the exact flattened size from the convolutional layers depends on input image
        # size and the number/configuration of convolutional and pooling layers.
        # Intermediate FC layers typically have a larger number of neurons (e.g., 256, 512).
        for i in range(num_fc_layers - 1): # All but the last FC layer (which is the output layer)
            classifier_layers.append(nn.LazyLinear(128)) # Example intermediate size for hidden layers
            classifier_layers.append(nn.ReLU())          # ReLU activation for non-linearity
            classifier_layers.append(nn.Dropout(0.5))     # Dropout for regularization to prevent overfitting
        
        # The final fully connected layer maps the features to the number of output classes.
        classifier_layers.append(nn.LazyLinear(num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """
        Defines the forward pass of the VGG model.

        Args:
            x (torch.Tensor): The input tensor (e.g., a batch of images).

        Returns:
            torch.Tensor: The output tensor containing logits for each class.
        """
        # Pass the input through the feature extraction (convolutional) layers
        x = self.features(x)
        # Flatten the output of the convolutional layers into a 1D vector for each sample
        # This prepares the data for input into the fully connected layers.
        x = x.view(x.size(0), -1) # x.size(0) is the batch size
        # Pass the flattened features through the classifier (fully connected) layers
        x = self.classifier(x)
        return x

# Instantiate the VGG model with desired parameters
# Here, we create a VGG-like model for MNIST:
# - 2 VGG blocks (`num_vgg_blocks=2`)
# - Each block has 1 convolutional layer (`convs_per_block=[1, 1]`)
# - The first block outputs 32 channels, the second outputs 64 channels (`channels_per_block=[32, 64]`)
# - The classifier has 2 fully connected layers (`num_fc_layers=2`), meaning one hidden layer and one output layer.
model = VGG(num_classes=10, num_vgg_blocks=2, convs_per_block=[1, 1], channels_per_block=[32, 64], num_fc_layers=2)

# Define the loss function and optimizer
# nn.CrossEntropyLoss is suitable for multi-class classification problems.
# torch.optim.Adam is an adaptive learning rate optimization algorithm.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
print("Starting training...")
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, labels in train_loader:
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)
        # Calculate the batch loss between predicted outputs and true labels
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad() # Clear any previously calculated gradients
        loss.backward()       # Compute gradients of the loss with respect to model parameters
        optimizer.step()      # Update model parameters based on the computed gradients

        total_loss += loss.item()
    
    # Calculate average loss for the current epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

print("Training finished.")

# Optional: Evaluate the model on test data
# This section assesses the model's performance on unseen data to check for generalization.
# Download and load test dataset (similar to training data loading)
test_data = datasets.MNIST(root='./data', train=False,
    transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
    shuffle=False)

model.eval() # Set the model to evaluation mode
# In evaluation mode, layers like Dropout behave differently (they are turned off).
correct = 0
total = 0
with torch.no_grad(): # Disable gradient calculation during evaluation
    # This saves memory and speeds up computation as gradients are not needed for inference.
    for inputs, labels in test_loader:
        outputs = model(inputs)
        # Get the index of the max log-probability (the predicted class)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # Accumulate total number of samples
        correct += (predicted == labels).sum().item() # Accumulate number of correct predictions

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")