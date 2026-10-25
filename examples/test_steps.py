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
lossFunction = MSELoss()

# Step 2: forward
out = net.forward(X)
print("forward shape:", out.shape, "(expected (4, 1))")

# Step 3: backward
lossBefore = lossFunction.forward(Y, out)
net.backward(lossFunction.backward())
print("dW shapes:", net.layers[0].dW.shape, net.layers[2].dW.shape, "(expected (2, 8) (8, 1))")

# Step 4: step
before = net.layers[0].W[0, 0]
net.step(0.1)
after = net.layers[0].W[0, 0]
print(f"W[0,0]: {before:.6f} -> {after:.6f} (should change slightly)")

lossAfter = lossFunction.forward(Y, net.forward(X))
print(f"loss: {lossBefore:.6f} -> {lossAfter:.6f} (should decrease)")
