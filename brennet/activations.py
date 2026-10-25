import numpy as np

class ReLU:
    def forward(self,Z):
        self.Z = Z #Save for backprop
        return np.maximum(0,Z)

    def backward(self, dA): 
        return dA * (self.Z > 0) #Derivative of ReLU

class Sigmoid:
    def forward(self,Z):
        self.Z = Z #Save for backprop
        self.A = 1 / (1 + np.exp(-Z)) #Save for backprop
        return self.A
  
    def backward(self, dA):
        sig = self.A
        return dA * sig * (1 - sig) #Derivative of sigmoid