import numpy as np
import multiprocessing
from functools import partial
from gradient_descent import gradient_descent
from loguru import logger
np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})

def generate_data(m=100, alpha=0.1, epochs=1000):
    # To introduce deterministic data - used for reproducibility
    np.random.seed(42)
    X = np.random.rand(m, 1)  # Feature (input)
    y = 4 * X + 2 + np.random.randn(m, 1) * 0.1  # Linear relationship with noise
    
    # Add bias term
    X_b = np.c_[np.ones((m, 1)), X]
    initial_thetas = [
        np.random.randn(2, 1),
        np.random.randn(2, 1),
        np.random.randn(2, 1),
        np.random.randn(2, 1)
    ]
    return X, X_b, y, alpha, epochs, m, initial_thetas

def run_multi_headed_gradient_descent(X, X_b, y, alpha, epochs, m, initial_thetas):
    # Create a pool of workers
    gradient_descent_with_provided_data = partial(gradient_descent, X_b, y, alpha, epochs, m)
    with multiprocessing.Pool(processes=len(initial_thetas)) as pool:
        results = pool.map(gradient_descent_with_provided_data, initial_thetas)
    
    # results is a list of tuples where first element is the initial 
    # theta value and the rest of the tuple is the converged theta
    converged_thetas = [item for _, item in results]
    for theta_start, theta_end in zip(initial_thetas, converged_thetas):
        logger.debug(f"Started with theta {theta_start}. Converged to {converged_thetas}")
    return converged_thetas

def main():
    X, X_b, y, alpha, epochs, m, initial_thetas = generate_data()
    logger.info(f"Kicking off gradient descent with values: M={m}, epochs={epochs}, alpha={alpha} with {len(initial_thetas)} processes")
    converged_thetas = run_multi_headed_gradient_descent(X, X_b, y, alpha, epochs, m, initial_thetas)
    logger.success(converged_thetas)

# Use multiprocessing to run gradient descent in parallel
if __name__ == "__main__":
    main()