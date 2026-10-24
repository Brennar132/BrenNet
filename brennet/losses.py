import numpy as np

class MSELoss:

    @staticmethod
    def MSE(Y, Y_hat):
        return np.mean((Y - Y_hat)**2)

    def forward(self, Y, Y_hat):
        self.Y = Y
        self.Y_hat = Y_hat
        return self.MSE(Y, Y_hat)
    
    def backward(self):
        n = self.Y.shape[0]
        return 2*(self.Y_hat - self.Y)/n #Derivative of MSE
