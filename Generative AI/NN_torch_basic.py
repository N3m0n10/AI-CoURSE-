"""Remaking the NN section binary classification with pytorch module!"""
import os # OS used for sampling error, in case you moved this or samples file
import numpy as np
import torch

if os.name == "nt": # NOTE: Windows uses backslashs
    data_path = "Neural_networks\Project_1\classification2.txt"
else: 
    data_path = "Neural_networks/Project_1/classification2.txt"


try:
    with open(data_path, "r") as input_file:
        training_data = input_file.readlines()
except FileNotFoundError:
    print(f"Error: Could not find {data_path}")
    print(f"Current working directory: {os.getcwd()}")

training_data = [line.strip().split(",") for line in training_data]
np.random.shuffle(training_data) # Shuffle data

input_data = np.array([[float(data[0]), float(data[1])] for data in training_data])
X = torch.tensor(input_data, dtype=torch.float32)
Y = torch.tensor([[int(data[2])] for data in training_data], dtype=torch.float32)

"""For simplicity the whole set will be used for training"""
class NeuralNetwork:
    def __init__(self,hidden_size,data,learning_rate=0.5,num_epochs=1000):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.target_data = data[1]
        self.input_data = data[0]
        self.input_size = self.input_data.shape[1]
        self.output_size = self.target_data.shape[1]
        self.model = self.simple_regression_model(self.input_size, self.hidden_size, self.output_size)
    
    def simple_regression_model(self,input_size, hidden_size, output_size):
        model = {
            'weights1': torch.randn(input_size, hidden_size, requires_grad=True),
            'bias1': torch.randn(1, hidden_size, requires_grad=True),
            'weights2': torch.randn(hidden_size, output_size, requires_grad=True),
            'bias2': torch.randn(1, output_size, requires_grad=True)
        }
        return model

    # Forward pass through the network
    def forward(self,model, inputs):
        h1 = inputs.mm(model['weights1']) + model['bias1']
        h1 = torch.relu(h1)
        output = h1.mm(model['weights2']) + model['bias2']
        # output = torch.sigmoid(output)  # Classification ## error taking raw logits, change if using normal BCE
        return output

    # Define a mean squared error (MSE) loss function
    def binary_cross_entropy_error(self,predictions, targets):
        loss_fn = torch.nn.BCEWithLogitsLoss()  # torch binary cross entropywith logits
        return loss_fn(predictions, targets)
    
    def training(self):

        # Neural Network structure
        model = self.model
        self.loss_list = []

        for epoch in range(self.num_epochs):
            # Forward pass
            predictions = self.forward(model, self.input_data)

            # Compute the loss
            loss = self.binary_cross_entropy_error(predictions, self.target_data)

            # Backpropagation
            loss.backward()

            # Update the model parameters using gradient descent
            with torch.no_grad():
                model['weights1'] -= self.learning_rate * model['weights1'].grad
                model['bias1'] -= self.learning_rate * model['bias1'].grad
                model['weights2'] -= self.learning_rate * model['weights2'].grad
                model['bias2'] -= self.learning_rate * model['bias2'].grad

            # Zero out gradients for the next iteration
            model['weights1'].grad.zero_()
            model['bias1'].grad.zero_()
            model['weights2'].grad.zero_()
            model['bias2'].grad.zero_()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item()}')

            self.loss_list.append(loss.item())

    def plots(self):
        import matplotlib.pyplot as plt

        # Loss plot
        plt.plot(np.linspace(0,self.num_epochs,self.num_epochs),self.loss_list,
                 label = f"learning rate:{self.learning_rate}\nhidden size: {self.hidden_size}")
        plt.title("Cross entropy loss by epoch!")
        plt.legend()
        plt.show()

        # Contour plot
        x1 = np.linspace(X[:,0].min(), X[:,0].max(), self.input_data.shape[0])
        x2 = np.linspace(X[:,1].min(), X[:,1].max(), self.input_data.shape[0])
        X1, X2 = np.meshgrid(x1, x2)

        grid = np.c_[X1.ravel(), X2.ravel()]
        grid_tensor = torch.tensor(grid, dtype=torch.float32)

        Z = self.forward(self.model, grid_tensor)
        Z = Z.detach().numpy().reshape(X1.shape)

        plt.contour(X1, X2, Z, levels=[0.5], cmap='RdBu')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Decision Boundary")
        plt.show()

    def save_results(self): # called in the jupyter lab doc yet TODO!
        return self.model

# Initialize the model
NN = NeuralNetwork(hidden_size=50, num_epochs=2000 , data=[X,Y])

# train the model
NN.training()

input("Plot results? [y/n]") == "y" and NN.plots()  # Plot when the eq is True 