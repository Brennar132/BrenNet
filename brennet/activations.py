import numpy as np

class ReLU:
    def forward(self,Z):
        self.Z = Z #Save for backprop
        return np.maximum(0,Z)

    def backward(self, dA): 
        return dA * (self.Z > 0) #Derivative of ReLU
