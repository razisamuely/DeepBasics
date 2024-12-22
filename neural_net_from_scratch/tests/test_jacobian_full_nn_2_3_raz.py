import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from neural_net_from_scratch.src.models.neural_network import DynamicNeuralNetwork
from neural_net_from_scratch.src.models.neural_network import ResidualBlock
from neural_net_from_scratch.src.models.neural_network import Layer
from neural_net_from_scratch.src.losses.sofmax_loss import SoftmaxLoss

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def get_layer_params(layer):
    if isinstance(layer, ResidualBlock):
        params = []
        for sublayer in layer.layers:
            params.append((sublayer.W.copy(), sublayer.b.copy()))
        return params
    else:
        return [(layer.W.copy(), layer.b.copy())]

def set_layer_params(layer, params):
    if isinstance(layer, ResidualBlock):
        for sublayer, param in zip(layer.layers, params):
            sublayer.W = param[0]
            sublayer.b = param[1]
    else:
        layer.W = params[0][0]
        layer.b = params[0][1]

def get_layer_gradients(layer):
    if isinstance(layer, ResidualBlock):
        grads = []
        for sublayer in layer.layers:
            grads.append((sublayer.dW, sublayer.db))
        return grads
    else:
        return [(layer.dW, layer.db)]

def get_layer_configs():
    return [
        {"type": "layer", "input_dim": 4, "output_dim": 8, "activation": "tanh"},
        {"type": "residual", "input_dim": 8, "output_dim": 8, "activation": "tanh"},
        {"type": "layer", "input_dim": 8, "output_dim": 3, "activation": "tanh"}
    ]

def generate_data():
    x = np.random.randn(4, 1) 
    y = np.array([[1, 0, 0]]).T 
    return x, y

def get_network_gradients(network, grad):
            grads = []
            for layer in reversed(network.layers):
                grad = layer.backward(grad)
                layer_grads = get_layer_gradients(layer)
                grads = layer_grads + grads
            return grads

def prepare_perturbation(params):
        return [(np.random.randn(*p[0].shape), np.random.randn(*p[1].shape)) 
                for p in params]

def test_full_network_gradient():
    layer_configs = get_layer_configs()
    network = DynamicNeuralNetwork(layer_configs)
    softmax = SoftmaxLoss()
    
    x, y = generate_data()
    
    def loss_function(params):
        current = x
        param_idx = 0
        for layer in network.layers:
            layer_params = params[param_idx:param_idx + (2 if isinstance(layer, ResidualBlock) else 1)]
            set_layer_params(layer, layer_params)
            current = layer.forward(current)
            param_idx += 2 if isinstance(layer, ResidualBlock) else 1
            
        probs = softmax.sofotmax(current)
        loss = softmax.loss(probs, y)        
        grad = softmax.loss_gradient(probs, y)
        
        grads = get_network_gradients(network, grad)
        return loss, grads
    
    current_params = []

    for layer in network.layers:
        current_params.extend(get_layer_params(layer))
    
    pertubation_noise = prepare_perturbation(current_params)
    
    epsilon = 0.1
    linear_errors = []
    quadratic_errors = []
    eps_values = []
    
    for i in range(20):
        eps = epsilon * (0.5 ** i)
        eps_values.append(eps)
        
        
        f_x, grads = loss_function(current_params)
        
        # Perturb wheights per layer
        perturbed_params = [(p[0] + eps * d[0], p[1] + eps * d[1]) 
                           for p, d in zip(current_params, pertubation_noise)]
        
        f_x_plus_eps, _ = loss_function(perturbed_params)
        
        dd = sum(np.sum(g[0] * d[0]) + np.sum(g[1] * d[1]) for g, d in zip(grads, pertubation_noise))
        
        linear_error = abs(f_x_plus_eps - f_x)
        quadratic_error = abs(f_x_plus_eps - f_x - eps * dd)
        
        linear_errors.append(linear_error)
        quadratic_errors.append(quadratic_error)
    
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
    plt.title('Full Network Gradient Test')
    plt.legend()
    plt.savefig('neural_net_from_scratch/artifacts/full_network_gradient_test_raz.png')
    plt.close()
    
    return linear_errors, quadratic_errors

if __name__ == "__main__":
    test_full_network_gradient()