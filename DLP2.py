import numpy as np

def fp(w, X):
    # Forward pass: matrix multiply weights (num_classes x num_features) with batch inputs (batch_size x num_features)
    # Return shape: (batch_size, num_classes)
    return np.dot(X, w.T)

def somax(r):
    # Softmax along classes axis (axis=1) for batch
    e = np.exp(r - np.max(r))  # stability trick
    return e / np.sum(e)

def cce(y_true, y_pred):
    # Average cross-entropy loss over batch
    clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
    loss = -np.sum(y_true * np.log(clipped)) / y_true.shape[0]
    return loss

def bpa(y_true, y_pred):
    # Gradient of loss wrt predictions
    return (y_pred - y_true) / y_true.shape[0]

def wtupdate(lr, X, grad):
    # Weight update: gradient is (batch_size x num_classes)
    # X is (batch_size x num_features)
    # We want to update weights shape (num_classes x num_features)
    return lr * np.dot(grad.T, X)  # shape: (num_classes, num_features)

def adjust_learning_rate(initial_lr, epoch, decay_rate=0.1, decay_epoch=10):
    # Simple step decay learning rate
    return initial_lr * (1 / (1 + decay_rate * (epoch // decay_epoch)))

# Example data (batch of 3 samples, 2 features each)
X = np.array([[1.0, 2.0], 
              [1.5, 2.5], 
              [2.0, 3.0]])  # shape (3, 2)

# Initial weights (3 classes, 2 features)
w = np.array([[0.2, -0.3], 
              [-0.5, 0.7], 
              [-0.2, 0.5]])  # shape (3, 2)

# One-hot true labels for batch of 3 samples, 3 classes
y_true = np.array([[0, 1, 0], 
                   [1, 0, 0], 
                   [0, 0, 1]])  # shape (3, 3)

initial_lr = 0.1
epoch = 20
lr = adjust_learning_rate(initial_lr, epoch)

# Forward pass
dotp = fp(w, X)         # shape (3, 3)
som = somax(dotp)       # softmax output, shape (3, 3)

# Compute loss
ccloss = cce(y_true, som)
print(f"Epoch {epoch}, Learning Rate: {lr:.4f}, Cross-Entropy Loss: {ccloss:.4f}")

# Backpropagation
grad = bpa(y_true, som)  # shape (3, 3)

# Weight update
update = wtupdate(lr, X, grad)
w -= update

print(f"Updated weights:\n{w}")

x_softmax = np.array([-2, -1, 0, 1, 2])
y_softmax = somax(x_softmax)
import matplotlib.pyplot as plt
plt.plot(x_softmax, y_softmax, 'o-')  # markers with line
plt.show()


#Program 2: Implement Categorical cross entropy Loss function with forward and backward pass
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability improvement
    return exp_x / np.sum(exp_x)

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)  # Avoid log(0)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def categorical_cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

def adjust_learning_rate(initial_lr, epoch, decay_rate=0.1, decay_epoch=10):
    return initial_lr * (1 / (1 + decay_rate * (epoch // decay_epoch)))

def forward_pass(X, weights):
    logits = np.dot(X, weights)
    y_pred = softmax(logits)
    return y_pred

def backward_pass(X, y_true, y_pred, learning_rate):
    gradient = categorical_cross_entropy_derivative(y_true, y_pred)
    weights_update = np.dot(X.T, gradient) / X.shape[0]
    return weights_update * learning_rate

# softmax input
x_softmax = np.array([-2, -1, 0, 1, 2])
y_softmax = softmax(x_softmax)
# Plot Softmax 
plt.figure(figsize=(6, 4))
plt.plot(x_softmax, y_softmax, marker='o', label='Softmax Output')
plt.title("Softmax Activation Function")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.grid()
plt.legend()
plt.show()
# Input features
X_sample = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]) 
#weights
weights = np.array([[0.2, -0.3, 0.5], [-0.5, 0.7, -0.2]])
# One-hot encoded labels
y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) 
# Forward pass
y_pred = forward_pass(X_sample, weights)
loss = categorical_cross_entropy(y_true, y_pred)
# Backward pass
learning_rate = 0.1
weight_update = backward_pass(X_sample, y_true, y_pred, learning_rate)
print("Predicted Probabilities:", y_pred)
print("Categorical Cross-Entropy Loss:", loss)
print("Gradient of Loss:", weight_update)
# Adjust learning rate
initial_lr = 0.1
epoch = 20
print("Adjusted Learning Rate:", adjust_learning_rate(initial_lr, epoch))