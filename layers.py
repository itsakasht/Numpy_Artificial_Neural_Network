import numpy as np

class Layer:
    def forward(self, inputs):
        pass

    def backward(self, doutput):
        pass

class LayerDense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = 0.1*np.random.randn(output_size, input_size) # Weights (output_size, input_size)
        self.biases = np.zeros((output_size, 1)) # Bias (output_size, 1) (column vector)

    def forward(self, inputs):
        self.input = inputs # inputs = (features, samples)
        self.output = self.weights @ inputs + self.biases # output (output_size, samples)
        return self.output

    def backward(self, doutput):
        self.dweights = doutput @ self.input.T
        self.dbiases = np.sum(doutput, axis=1, keepdims=True)
        self.dinputs = self.weights.T @ doutput # dinputs = (input_size, samples)
        return self.dinputs
    
    def gradient(self):
        return self.dweights, self.dbiases

class Dropout(Layer):
    def __init__(self, rate=0.5):
        """
        rate: dropout probability (fraction of neurons dropped)
        keep_prob = 1 - rate
        """
        self.rate = rate
        self.keep_prob = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            # Create mask: 1 with probability keep_prob, else 0
            self.mask = (np.random.rand(*inputs.shape) < self.keep_prob) / self.keep_prob
            self.output = inputs * self.mask
        else:
            # During inference, no dropout applied
            self.output = inputs
        return self.output

    def backward(self, doutput):
        # Gradient passes only through the active neurons
        self.dinputs = doutput * self.mask
        return self.dinputs


class ActivationReLU(Layer):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output # output (output_size, samples)

    def backward(self, doutput):
        self.dinputs = doutput * (self.output > 0)
        return self.dinputs

    
class ActivationTanh(Layer):
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output # output (output_size, samples)

    def backward(self, doutput):
        self.dinputs = doutput * (1 - self.output ** 2)
        return self.dinputs

class ActivationSoftmaxLossCrossEntropy(Layer):
    def forward(self, inputs):
        self.input = inputs

        # Softmax activation (numerical stability)
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return self.output # output (classes, samples)
    
    def loss(self, true_labels):
        self.true_labels = true_labels

        # Cross-entropy loss
        # true_labels are one-hot: shape (classes, samples)
        correct_probs = np.sum(self.output * self.true_labels, axis=0)
        
        # Loss per sample
        losses = -np.log(correct_probs + 1e-9)
        return np.mean(losses)

    def backward(self, y_true):
        self.true_labels = y_true
        samples = self.output.shape[1]
        self.dinputs = (self.output - self.true_labels) / samples
        return self.dinputs


class Loss:
    def calculate(self, output, y_true):
        sample_losses = self.forward(output, y_true)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def derivative(self):
        pass

class CrossEntropyLoss(Loss):
    def forward(self, output, y_true):
        sample_losses = np.sum(y_true * np.log(output + 1e-9), axis=0) # One-hot encoded y_true
        return -sample_losses
    
    def backward(self, output, y_true):
        pass

class GradientDescent:
    def __init__(self, learning_rate):
        self.lr = learning_rate
    
    def update(self, layer):
        layer.weights -= self.lr * layer.dweights
        layer.biases -= self.lr * layer.dbiases

class MomentumGradientDescent:
    def __init__(self, learning_rate, beta):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}
    
    def update(self, layer):
        if layer not in self.velocities:
            self.velocities[layer] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        v = self.velocities[layer]
        v['weights'] = self.beta * v['weights'] - self.lr * layer.dweights
        v['biases'] = self.beta * v['biases'] - self.lr * layer.dbiases

        layer.weights += v['weights']
        layer.biases += v['biases']

class NestrovGradientDescent:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer):
        if layer not in self.velocities:
            self.velocities[layer] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        v = self.velocities[layer]

        # lookahead position
        lookahead_w = layer.weights + self.momentum * v['weights']
        lookahead_b = layer.biases + self.momentum * v['biases']

        # compute gradients at lookahead (assumes layer gradients are ready)
        v['weights'] = self.momentum * v['weights'] - self.lr * layer.dweights
        v['biases'] = self.momentum * v['biases'] - self.lr * layer.dbiases

        layer.weights += v['weights']
        layer.biases += v['biases']

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, layer):
        if layer not in self.m:
            self.m[layer] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.v[layer] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }

        # Time step
        self.t += 1

        # Moving averages of gradient
        self.m[layer]['weights'] = self.beta1 * self.m[layer]['weights'] + (1 - self.beta1) * layer.dweights
        self.m[layer]['biases'] = self.beta1 * self.m[layer]['biases'] + (1 - self.beta1) * layer.dbiases

        # Moving averages of squared gradient
        self.v[layer]['weights'] = self.beta2 * self.v[layer]['weights'] + (1 - self.beta2) * (layer.dweights ** 2)
        self.v[layer]['biases'] = self.beta2 * self.v[layer]['biases'] + (1 - self.beta2) * (layer.dbiases ** 2)

        # Bias correction
        m_hat_w = self.m[layer]['weights'] / (1 - self.beta1 ** self.t)
        m_hat_b = self.m[layer]['biases'] / (1 - self.beta1 ** self.t)
        v_hat_w = self.v[layer]['weights'] / (1 - self.beta2 ** self.t)
        v_hat_b = self.v[layer]['biases'] / (1 - self.beta2 ** self.t)

        # Parameter update
        layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
        layer.biases  -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)



def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T