class Layer:
    #abstract baseclass

    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, out_error, lr):
        pass




