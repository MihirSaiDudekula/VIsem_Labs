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

# First Lab Program: Implement training rate and all activation functions.
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability improvement
    return exp_x / np.sum(exp_x)

def adjust_learning_rate(initial_lr, epoch, decay_rate=0.1, decay_epoch=10):
    return initial_lr * (1 / (1 + decay_rate * (epoch // decay_epoch)))

# input values
x = np.linspace(-5, 5, 100)

# Plot activation functions
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

functions = [(sigmoid, "Sigmoid"), (tanh, "Tanh"), (relu, "ReLU"), (leaky_relu, "Leaky ReLU")]

for i, (func, title) in enumerate(functions):
    y = func(x)
    axes[i].plot(x, y)
    axes[i].set_title(title)
    axes[i].grid()

# Softmax applied to vectors
x_softmax = np.linspace(-2, 2, 5)  
y_softmax = softmax(x_softmax)
axes[4].plot(x_softmax, y_softmax, marker='o')
axes[4].set_title("Softmax")
axes[4].grid()
# Hide the last empty subplot
axes[5].axis("off")
plt.tight_layout()
plt.show()
x_sample = np.array([-2, -1, 0, 1, 2])
print("Sigmoid:", sigmoid(x_sample))
print("Tanh:", tanh(x_sample))
print("ReLU:", relu(x_sample))
print("Leaky ReLU:", leaky_relu(x_sample))
print("Softmax:", softmax(x_sample))

# training rate adjustment
initial_lr = 0.1
epoch = 20
print("Adjusted Learning Rate:", adjust_learning_rate(initial_lr, epoch))