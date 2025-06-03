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
