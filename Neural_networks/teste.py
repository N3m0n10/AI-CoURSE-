import numpy as np

def generate_circle_data(n_samples=300, noise=0.1, inner_radius=0.5, outer_radius=1.0):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    theta_out = 2 * np.pi * np.random.rand(n_samples_out)
    r_out = outer_radius + noise * np.random.randn(n_samples_out)
    X_out = np.stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)], axis=1)
    Y_out = np.zeros((n_samples_out, 1))

    theta_in = 2 * np.pi * np.random.rand(n_samples_in)
    r_in = inner_radius + noise * np.random.randn(n_samples_in)
    X_in = np.stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)], axis=1)
    Y_in = np.ones((n_samples_in, 1))

    X = np.vstack([X_in, X_out])
    Y = np.vstack([Y_in, Y_out])

    indices = np.random.permutation(len(X))
    return X[indices], Y[indices]

X, Y = generate_circle_data()
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

def xavier_init(out_dim, in_dim):
    return np.random.randn(out_dim, in_dim) * np.sqrt(1.0 / in_dim)

def forward(X, W1, W2):
    Z1 = np.dot(X, W1.T)
    A1 = sigmoid(Z1)
    A1_bias = np.hstack([np.ones((A1.shape[0], 1)), A1])
    Z2 = np.dot(A1_bias, W2.T)
    A2 = sigmoid(Z2)
    return A1, A2, A1_bias

def compute_loss(Y, Y_hat):
    return -np.mean(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8))

def backward(X, Y, W1, W2, A1, A1_bias, A2, lr=0.1):
    m = X.shape[0]
    dZ2 = A2 - Y  # (m, 1)
    dW2 = (1/m) * np.dot(dZ2.T, A1_bias)

    dA1 = np.dot(dZ2, W2[:, 1:])  # skip bias
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = (1/m) * np.dot(dZ1.T, X)

    W1 -= lr * dW1
    W2 -= lr * dW2
    return W1, W2

np.random.seed(42)
input_dim = 2
hidden_dim = 6

W1 = xavier_init(hidden_dim, input_dim)
W2 = xavier_init(1, hidden_dim + 1)  # +1 for bias

for epoch in range(10000):
    A1, A2, A1_bias = forward(X, W1, W2)
    loss = compute_loss(Y, A2)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} — Loss: {loss:.4f}")
    W1, W2 = backward(X, Y, W1, W2, A1, A1_bias, A2)

_, A2_final, _ = forward(X, W1, W2)
preds = (A2_final > 0.5).astype(int)
accuracy = np.mean(preds == Y)
print("Final Accuracy:", accuracy)
print("predictions:", preds.flatten())
print(preds.flatten() - Y.flatten())

