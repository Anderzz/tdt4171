class Layer:
    #abstract baseclass

    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagation(self, input):
        pass

    def backward_propagation(self, out_error, lr):
        pass




