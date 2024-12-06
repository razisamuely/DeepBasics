import numpy as np
from neural_network import NeuralNetwork

class ResidualNetwork(NeuralNetwork):
    def __init__(self, block_layer_sizes, activation_function=None, last_activation_function=None):
        """
        Initialize a residual network.
        The network consists of multiple layers as defined by block_layer_sizes.
        After passing through all layers, the input is added to the final output (residual connection).

        Args:
            block_layer_sizes (list): Layer sizes for the block.
                                      Must have block_layer_sizes[0] == block_layer_sizes[-1].
                                      e.g., [input_dim, hidden_dim, ..., hidden_dim, output_dim]
            activation_function (callable): Activation function for hidden layers.
            last_activation_function (callable): Activation function for the output layer.
        """
        super().__init__(block_layer_sizes, activation_function, last_activation_function)

    def forward(self, X):
        """
        Forward pass through the residual network.
        We apply a series of linear transformations followed by activations. After the last layer,
        we add the original input to the output (residual connection), then apply the last activation (if any).

        Returns:
            ndarray: The final output after the residual connection.
        """
        self.outputs = []
        self.pre_activations = []

        # Store the original input for the residual connection (without bias augmentation)
        original_input = X.copy()

        # Augment input with bias
        X = np.vstack([X, np.ones(X.shape[1])])
        if self.grad:
            self.outputs.append(X)

        # Forward through each layer
        for i in range(len(self.weights)):
            # Compute pre-activation
            Z = self.weights[i] @ X

            # If not the last layer, apply the hidden activation and augment with bias
            if i < len(self.weights) - 1:
                if self.activation_function is not None:
                    X = self.activation_function.activation(Z)
                else:
                    X = Z
                X = np.vstack([X, np.ones(X.shape[1])])  # add bias
            else:
                # Last layer (before residual addition)
                if self.activation_function is not None:
                    X = self.activation_function.activation(Z)
                else:
                    X = Z

            if self.grad:
                self.pre_activations.append(Z)
                self.outputs.append(X)

        # Add the residual connection: output + original_input
        # Dimensions must match: block_layer_sizes[-1] == block_layer_sizes[0]
        X = X + original_input

        # If there's a last activation function, apply it now
        if self.last_activation_function is not None:
            X = self.last_activation_function.activation(X)
            if self.grad:
                self.outputs[-1] = X  # Replace the last stored output
        else:
            if self.grad:
                self.outputs[-1] = X  # Replace the last stored output with residual added

        return X

    def backward(self, x, y):
        """
        Backward pass through the residual network.

        Steps:
        1. Compute error = final_output - y
        2. If last_activation_function exists, multiply error by its derivative wrt pre-activation of last layer.
        3. Propagate the error back through each layer:
           - For each layer, compute the gradient w.r.t. weights.
           - Move error backward by multiplying with W (ignoring bias).
           - Apply derivative of activation function for hidden layers.
        4. The residual connection means final_output = layer_output + original_input.
           Since input is not a parameter, we just pass the error backward through the layers.
        """
        assert self.grad and len(self.outputs) != 0 and len(self.pre_activations) != 0, \
            "Need grad=True and a forward pass before backprop."

        gradients = []
        final_output = self.outputs[-1]

        # Compute initial error
        error = final_output - y

        # If there's a last activation function, apply its derivative
        if self.last_activation_function is not None:
            error *= self.last_activation_function.derivative(self.pre_activations[-1])

        # We have a chain of layers in self.weights
        # We will backprop through them in reverse order
        for i in reversed(range(len(self.weights))):
            # Output of the (i)-th layer was stored in self.outputs[i]
            # Pre-activation of the (i)-th layer was stored in self.pre_activations[i]

            # If not the last layer, apply derivative of the hidden activation function
            # If the last layer was just handled by last_activation_function above, we still
            # might need to apply the hidden activation derivative (because the last layer also had an activation)
            if i < len(self.weights) - 1 or (
                    self.activation_function is not None and self.last_activation_function is None):
                error *= self.activation_function.derivative(self.pre_activations[i])

            # Compute gradient for W[i]
            # The output feeding into layer i was self.outputs[i - 1] if i > 0, else self.outputs[0] is input
            # Actually, self.outputs[i] is the output *after* that layer. The input to layer i is self.outputs[i-1].
            # For i=0, the input was the augmented initial input stored in self.outputs[0].
            # For i>0, the input to layer i is self.outputs[i-1].
            # However, note that self.outputs[i] corresponds to X after layer i, and self.pre_activations[i] for Z[i].
            # The input to layer i is actually self.outputs[i-1], except for i=0 where input is self.outputs[0].
            # But we must remember how we stored outputs:
            # After forward: self.outputs[0] = augmented input X
            #                self.outputs[1] = output after layer 1
            #                ...
            # So for layer i, input = self.outputs[i-1] (with i-1 >= 0), and for i=0 input = self.outputs[0] is the input.
            # Actually, we stored every step along the way, so input to layer i is self.outputs[i-1],
            # but i=0 means input is at index 0 of self.outputs: the augmented input X.

            layer_input = self.outputs[i - 1] if i > 0 else self.outputs[0]
            grad_w = error @ layer_input.T / x.shape[1]
            gradients.insert(0, grad_w)

            # Propagate error backward (if not the first layer)
            # Remove the bias component from weights before backprop
            if i > 0:
                W_no_bias = self.weights[i][:, :-1]
                error = W_no_bias.T @ error

        # Clear stored values
        self.outputs = []
        self.pre_activations = []

        return gradients
