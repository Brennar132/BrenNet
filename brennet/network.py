
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        """
        Performs forward propagation through the network.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, d):
        """
        Performs backward propagation.
        """""
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def step(self, learningRate):
        """
        Performs a single optimization step.
        """
        for layer in self.layers:
            if hasattr(layer, 'W'): # Check if the layer has weights 
                layer.W -= learningRate * layer.dW 
                layer.b -= learningRate * layer.db

    def train(self, X, y, loss_fn, epochs, learningRate):
        """
        Trains the network using the provided data.
        """
        for i in range(epochs):
            predictions = self.forward(X)  # Forward pass
            loss = loss_fn.forward(y,predictions)  # Compute loss
            derivative = loss_fn.backward()  # Compute derivative of loss
            self.backward(derivative)  # Backward pass
            self.step(learningRate)  # Update weights and biases
            
            if i % 1000 == 0:
              print(f"epoch {i:5d}  loss {loss:.6f}")


         