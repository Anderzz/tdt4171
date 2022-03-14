class Layer:
    #abstract baseclass

    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        pass

    def backward(self, out_error, lr=0.01):
        pass




