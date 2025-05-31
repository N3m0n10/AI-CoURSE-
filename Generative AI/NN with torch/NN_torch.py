"Going further with pytorch Neural Networks"
"Using the NN module for a regression problem"
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleFeedForward(nn.Module):
    def __init__(self, input_tensor, hidden_size, output_tensor,loss_function):
        super(SimpleFeedForward, self).__init__()
        # Define the layers and their corresponding activations
        self.loss_function = loss_function
        self.input_size = input_tensor.size(-1)
        self.output_size = output_tensor.size(-1)
        self.fc1 = nn.Linear(self.input_size, hidden_size) # Fully connected layer
        self.relu = nn.ReLU() # Activation function
        self.fc2 = nn.Linear(hidden_size, self.output_size) # Fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def loss_(self,output_):  # Prepare loss, needed due to different args
        if self.loss_function == mse_loss_with_l2:
            loss = self.loss_function(output_, target_tensor, net, 0.001)
        else:
            loss = self.loss_function(output_, target_tensor)
        return loss
    
    def train(self,epochs_):
        # Define a loss and an optimizer
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)

        for i in range(epochs_):
            # Forward pass
            output = net(input_tensor)

            # Calculate the loss
            loss = self.loss_(output)

            # Backward pass and optimization
            optimizer.zero_grad() # Zero the gradients
            loss.backward()

            # Compute the gradient of the loss with respect to model parameters
            optimizer.step()
            # Update the model parameters
    
def mse(predictions, target):
    # return ((target - predictions) ** 2).mean()
    return ((target - predictions) ** 2).sum() / target.numel()

def mse_loss_with_l2(predictions, targets, model, lambda_l2=0.01):
    mse = ((predictions - targets) ** 2).mean()

    # Add L2 regularization term
    l2_regularization = 0.0
    for param in model.parameters():
        l2_regularization += param.norm(2)
    
    # L2 norm of each parameter
    # final loss by adding the MSE loss and regularization term
    return mse + lambda_l2 * l2_regularization
    
input_size = 3
hidden_size = 5
output_size = 2
learning_rate = 0.01
epochs = 100

# Generate dummy input tensor
input_tensor = torch.randn((1, input_size))

# Assume we have some target tensor for training
target_tensor = torch.randn((1, output_size))

#criterion = nn.MSELoss()
#criterion = mse        #NOTE: you can make your loss function
criterion = mse_loss_with_l2

# Initialize the network
net = SimpleFeedForward(input_tensor, hidden_size, target_tensor,loss_function=criterion)
net.train(epochs)

    
print(f"Updated Output after {epochs} steps of optimization:", net(input_tensor))
print("Desired Output:", target_tensor)