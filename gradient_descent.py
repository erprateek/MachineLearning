# Function to run gradient descent with a given initial parameter
import numpy as np

def gradient_descent(X_b, y, alpha, epochs, m, theta):
    provided_theta = theta    
    for epoch in range(epochs):
        y_pred = X_b.dot(theta)  # Predictions
        error = y_pred - y  # Compute error
        gradients = (1/m) * X_b.T.dot(error)  # Compute gradient
        theta -= alpha * gradients  # Update parameters
    return (provided_theta, theta.flatten())  # Return final parameters
