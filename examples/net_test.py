import numpy as np
from brennet import Network, constructDense, ReLU, Sigmoid, MSELoss

np.random.seed(0)

# XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
Y = np.array([[0],[1],[1],[0]], dtype=float)

net = Network([
    constructDense(2, 8),
    ReLU(),
    constructDense(8, 1),
    Sigmoid(),
])

net.train(X, Y, MSELoss(), epochs=5001, learningRate=0.1)

print("\nPredictions:")
for x, y, p in zip(X, Y, net.predict(X)):
    print(f"  {x} -> {p[0]:.3f}  (target: {y[0]:.0f})")
