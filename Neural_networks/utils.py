def gradient_descent(X, Y, W1, W2, learning_rate=0.01, epochs=1000):
    for epoch in range(epochs):
        # Forward propagation
        A1, A2 = foward_prop(X, W1, W2)

        # Compute cost
        cost = J(A2, Y, W1, W2)
        
        # Backward propagation
        dW1, dW2 = backward_prop(X, Y, A1, A2, W2)

        # Update weights
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")

    return W1, W2

def backward_prop(X, Y, A1, A2, W2):
    # Calculate the gradients for W2
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2)

    # Calculate the gradients for W1
    dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1)

    return dW1, dW2