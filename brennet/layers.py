import numpy as np

class constructDense: #Dense as in fully connected
    
    def __init__(self, inputNeurons, outputNeurons):
        dev = np.sqrt(2/inputNeurons)
        self.W  = np.random.randn(inputNeurons, outputNeurons)*dev
        self.b = np.zeros((1, outputNeurons))

    def forward(self, X):
        """ Z = xW + b"""
        self.X = X #Save for backprop
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.X.T @ dZ  #Derivative of the loss with respect to W
        self.db = np.sum(dZ, axis=0, keepdims=True) #Derivative
        dX = dZ @ self.W.T #Derivative of the loss with respect to X (before W is updated)
        return dX
