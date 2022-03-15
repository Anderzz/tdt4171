import numpy as np

from network import Network
from fclayer import Dense
from activation_layer import ActivationLayer
from activation_functions import tanh, dtanh
from loss import mse, dmse

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
model = Network()
model.add(Dense(2, 5))
model.add(ActivationLayer(tanh, dtanh))
model.add(Dense(5, 3))
model.add(ActivationLayer(tanh, dtanh))
model.add(Dense(3, 1))
model.add(ActivationLayer(tanh, dtanh))

# train
model.set_loss(mse, dmse)
model.fit(x_train, y_train, epochs=1000, lr=0.1)

# test
out = model.predict(x_train)
print(out)