import numpy as np
import tensorflow as tf
import tf_quant_finance as tff

# Define the optimization problem
def portfolio_optimization(cov_matrix: np.array, expected_returns: np.array)-> np.array:
    num_assets = len(expected_returns)
    P = tf.constant(cov_matrix, dtype=tf.float64)
    q = tf.constant(np.zeros(num_assets), dtype=tf.float64)
    G = tf.constant(np.diag(-np.ones(num_assets)), dtype=tf.float64)
    h = tf.constant(np.zeros(num_assets), dtype=tf.float64)
    A = tf.constant(np.ones((1, num_assets)), dtype=tf.float64)
    b = tf.constant(1.0, dtype=tf.float64)

    # Solve the quadratic programming problem
    sol = tff.math.qp.solve_qp(P, q, G, h, A, b)
    return sol.x.numpy()

if __name__ == "__main__":
    # Define the covariance matrix
    cov_matrix = np.array([[1.0, 0.3, 0.4], [0.3, 1.0, 0.5], [0.4, 0.5, 1.0]])

    # Define the expected returns
    expected_returns = np.array([0.1, 0.2, 0.15])
    
    # Run the optimization
    optimal_weights = portfolio_optimization(cov_matrix, expected_returns)
    print("Optimal Weights:", optimal_weights)
