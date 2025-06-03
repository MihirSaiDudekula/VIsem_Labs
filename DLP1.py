import numpy as np
import matplotlib.pyplot as plt

# 1. Generate x values
def get_x():
    return np.linspace(-10, 10, 100)

# 2. Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

# 3. Plot function
def plot_activation(x, y):
    plt.plot(x, y)
    plt.title("Activation Function")
    plt.grid(True)
    plt.show()

# 4. Use everything together
x = get_x()
ys = softmax(x)  # change to sigmoid(x), relu(x), etc. as needed
plot_activation(x, y)
