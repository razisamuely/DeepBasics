import numpy as np
import matplotlib.pyplot as plt
from neural_net_from_scratch.src.models.neural_network_raz import DynamicNeuralNetwork

def test_jacobian():
    layer_dims = [784,10]  
    activations = ['tanh']
    network = DynamicNeuralNetwork(layer_dims=layer_dims, activations=activations)

    x = np.random.randn(784, 1) 
    d = np.random.randn(784, 1)
    epsilon = 0.2

    linear_errors = []
    quadratic_errors = []
    eps_values = []

    for i in range(20):
        eps = epsilon * (np.power(0.5, i))
        eps_values.append(eps)

        f_x = network.forward(x)
        x_plus_eps_d = x + eps * d
        f_x_plus_eps_d = network.forward(x_plus_eps_d)
        
        activation_deriv = network.layers[0].activation_derivative(network.layers[0].z)
        JacMV = activation_deriv * (network.layers[0].W @ (eps * d))

        val_linear = np.linalg.norm(f_x_plus_eps_d - f_x)
        val_quad = np.linalg.norm(f_x_plus_eps_d - f_x - JacMV)

        linear_errors.append(val_linear)
        quadratic_errors.append(val_quad)

    plt.figure(figsize=(10, 6))
    plt.loglog(eps_values, linear_errors, 'b.-', label='Linear Error')
    plt.loglog(eps_values, quadratic_errors, 'r.-', label='Quadratic Error')
    
    ref_eps = np.array([eps_values[0], eps_values[-1]])
    ref_linear = ref_eps * linear_errors[0]/eps_values[0]
    ref_quad = ref_eps**2 * quadratic_errors[0]/eps_values[0]**2
    plt.loglog(ref_eps, ref_linear, 'b--', alpha=0.5, label='O(ε)')
    plt.loglog(ref_eps, ref_quad, 'r--', alpha=0.5, label='O(ε²)')

    plt.grid(True)
    plt.xlabel('ε')
    plt.ylabel('Error')
    plt.title('Jacobian Test Convergence')
    plt.legend()
    
    plt.savefig('neural_net_from_scratch/artifacts/jacobian_test.png')
    plt.close()

    return linear_errors, quadratic_errors

def test_convergence():
    linear_errors, quadratic_errors = test_jacobian()
    
    linear_ratios = np.array(linear_errors[:-1]) / np.array(linear_errors[1:])
    quad_ratios = np.array(quadratic_errors[:-1]) / np.array(quadratic_errors[1:])
    
    assert np.allclose(linear_ratios, 2, rtol=0.5), "Linear is not decrease by factor of 2"
    assert np.allclose(quad_ratios, 4, rtol=0.5), "Quadratic is not decrease by factor of 4"