class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.dloss = None
    
    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss, dloss):
        self.loss = loss
        self.dloss = dloss

    def predict(self, input):
        samples = len(input)
        result = []

        for i in range(samples):
            #forward pass
            output = input[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, lr):
        samples = len(x_train)

        #train loop
        for epoch in range(epochs):
            err = 0
            for i in range(samples):
                #forward pass
                output = x_train[i]
                for layer in self.layers:
                    output = layer.forward(output)

                err += self.loss(y_train[i], output)

                #backward pass
                error = self.dloss(y_train[i], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, lr)
            
            err /= samples
            if epoch % 10 == 0:
                print(f"epoch {epoch+1}/{epochs}  error {err}")

