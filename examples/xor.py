import numpy as np
from brennet import constructDense, ReLU, MSELoss

np.random.seed(0)

# XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([[0],[1],[1],[0]], dtype=float)

layer1 = constructDense(2, 8)
activation = ReLU()
layer2 = constructDense(8, 1)
lossFunction = MSELoss()
learningRate = 0.1

for epoch in range(5001):
    # Forward
    Y_hat = layer2.forward(activation.forward(layer1.forward(X)))
    loss = lossFunction.forward(Y, Y_hat)

    # Backward
    d = lossFunction.backward()
    d = layer2.backward(d, learningRate)
    d = activation.backward(d)
    layer1.backward(d, learningRate)

    if epoch % 1000 == 0:
        print(f"epoch {epoch:5d}  loss {loss:.6f}")

print("\nPredictions:")
Y_hat = layer2.forward(activation.forward(layer1.forward(X)))
for x, y, p in zip(X, Y, Y_hat):
    print(f"  {x} -> {p[0]:.3f}  (target: {y[0]:.0f})")
