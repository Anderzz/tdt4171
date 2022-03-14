from layer import Layer

class ActivationLayer(Layer):

    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_propagation(self, out_error, lr):
        return self.dactivation(self.input) * out_error

