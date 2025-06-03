import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class ANN:
    def __init__(self):
        self.w1 = np.array([[0.003, -0.008,  0.004,  0.001],
                            [-0.007,  0.002,  0.005, -0.006]])
        self.b1 = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.w2 = np.array([[ 0.002, -0.001,  0.004],
                            [ 0.006,  0.003, -0.002],
                            [-0.001,  0.007,  0.002],
                            [ 0.005, -0.003,  0.001]])
        self.b2 = np.array([[0.0, 0.0, 0.0]])
        self.lr = 0.1

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1  # shape: (batch_size, 4)
        self.a1 = relu(self.z1)                 # shape: (batch_size, 4)
        self.z2 = np.dot(self.a1, self.w2) + self.b2  # shape: (batch_size, 3)
        self.y_pred = softmax(self.z2)               # shape: (batch_size, 3)
        return self.y_pred

    def backward(self, x, y_true):
        m = x.shape[0]  # batch size

        dZ2 = self.y_pred - y_true                 # shape: (batch_size, 3)
        dW2 = np.dot(self.a1.T, dZ2) / m           # shape: (4, 3)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.w2.T) * relu_derivative(self.z1)  # shape: (batch_size, 4)
        dW1 = np.dot(x.T, dZ1) / m                                # shape: (2, 4)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.w2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, x, y_true, epochs=100):
        for i in range(epochs):
            y_pred = self.forward(x)
            loss = cross_entropy_loss(y_true, y_pred)
            self.backward(x, y_true)
            if i % 10 == 0:
                print(f"Epoch {i}: Loss = {loss:.4f}")

# Example data (2 input features, 3 output classes, one-hot encoded)
X = np.array([[0.5, 0.2],
              [0.9, 0.7],
              [0.4, 0.5]])

Y = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

model = ANN()
model.train(X, Y, epochs=100)