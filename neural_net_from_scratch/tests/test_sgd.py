import numpy as np
from neural_net_from_scratch.src.optimizers.sgd import SGD
from neural_net_from_scratch.src.losses.least_squares import LeastSquaresLoss
import matplotlib.pyplot as plt 

N_SAMPLES = 100
UPPER_BOUND = 10
N_FEATURES = 2
BATCH_SIZE = 100
N_ITERATIONS = 1500
LEARNING_RATE = 1e-6
NOISE_SCALE = 10

def generate_linear_regression_data():
    np.random.seed(42)
    y =  np.array([np.array([i + np.random.normal(loc = 0, scale =3 )]) for i in range(0,N_SAMPLES)]).squeeze()
    X =  np.array(np.array([(1,i + np.random.normal(loc = 0, scale =3 )) for i in range(0,N_SAMPLES)]))
    w_init = np.zeros(N_FEATURES) 

    return X, y, w_init

def get_analytical_solution(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def plot_analytical_solution_vs_sgd(X, y, w_sgd, w_analytical, analytical_weights, sgd_weights):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,1:], y, s=10)
    plt.plot(X[:,1:], X @ w_sgd, color='red', linewidth=2, label='SGD')
    plt.plot(X[:,1:], X @ w_analytical, color='blue', linewidth=2, label='Analytical')
    plt.grid(True)
    plt.title(f"""SGD  vs Analytical\nSGD solution: {sgd_weights}\nanalytical solution: {analytical_weights}""")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('neural_net_from_scratch/artifacts/sgd_vs_analytical.png')


def test_sgd_optimization():
    X, y, w_init = generate_linear_regression_data()
    w_analytical = get_analytical_solution(X, y)
    
    model = LeastSquaresLoss()
    optimizer = SGD(
        model=model,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        number_of_iteration=N_ITERATIONS
    )
    
    w_sgd = optimizer.run(X, y, w_init)

    plot_analytical_solution_vs_sgd(X, y, w_sgd, w_analytical, w_analytical, w_sgd)
    
if __name__ == "__main__":
    print("Testing SGD optimization...")
    test_sgd_optimization()
