import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def cross_entropy_loss(logits, targets):
    # Compute the log softmax of the logits
    log_softmax = torch.log_softmax(logits, dim=1)
    # Gather the log probabilities corresponding to the true class labels
    loss = -torch.sum(log_softmax * targets) / targets.size(0)
    return loss
    
# Create the model
model = SimpleMLP(input_size=784, hidden_size=500, num_classes=10)
print(model)

#criterion = nn.CrossEntropyLoss() 
criterion = cross_entropy_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy data
inputs = torch.randn(64, 784) # 64 samples, each of size 784
#labels = torch.randint(0, 10, (64,)) # 64 labels ranging from 0 to 9
labels = torch.zeros(64,10)
for res in labels:   # NOTE: one-hot encoding for custom criterion
    n = torch.randint(0,10,(1,))
    res[n] = 1
#print(labels)  # 64 labels of len(10) lists, (classes)!

for _ in range(50):
    # Forward pass
    outputs = model(inputs) # before activation [logits]

    #loss
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Loss: {loss.item()}")
#TODO: use MSINT