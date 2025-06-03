

---

## 🧩 **Setup**

Assume:

* Input feature vector: `X = [[1.0, 2.0]]` (1 sample, 2 features)
* Weights:

  $$
  W = \begin{bmatrix} 0.2 & -0.3 & 0.5 \\ -0.5 & 0.7 & -0.2 \end{bmatrix}
  $$

  (2 features → 3 classes)
* True label (one-hot): `y_true = [[0, 1, 0]]` → class 1 is correct

---

## ✅ **Step-by-step Execution (Forward + Backward)**

---

### **1. Forward Pass — Compute Logits**

Logits = `X · W`

$$
\begin{bmatrix} 1.0 & 2.0 \end{bmatrix}
\cdot
\begin{bmatrix} 0.2 & -0.3 & 0.5 \\ -0.5 & 0.7 & -0.2 \end{bmatrix}
=
\begin{bmatrix} (1*0.2 + 2*(-0.5)) & (1*(-0.3) + 2*0.7) & (1*0.5 + 2*(-0.2)) \end{bmatrix}
$$

$$
= \begin{bmatrix} 0.2 - 1.0 & -0.3 + 1.4 & 0.5 - 0.4 \end{bmatrix} = \begin{bmatrix} -0.8 & 1.1 & 0.1 \end{bmatrix}
$$

---

### **2. Apply Softmax**

Let’s apply softmax to logits `[-0.8, 1.1, 0.1]`.

First, subtract max for numerical stability:

$$
\text{logits} - \max = [-0.8, 1.1, 0.1] - 1.1 = [-1.9, 0, -1.0]
$$

Exponentiate:

$$
\exp = [e^{-1.9}, e^0, e^{-1.0}] ≈ [0.149, 1.0, 0.368]
$$

Sum:

$$
\text{sum} = 0.149 + 1.0 + 0.368 = 1.517
$$

Now divide:

$$
\text{softmax} = \left[ \frac{0.149}{1.517}, \frac{1.0}{1.517}, \frac{0.368}{1.517} \right] ≈ [0.098, 0.659, 0.242]
$$

This is our **predicted probabilities**:

$$
y_{\text{pred}} = [0.098, 0.659, 0.242]
$$

---

### **3. Categorical Cross-Entropy Loss**

Recall: true label = `[0, 1, 0]` (i.e., class 1 is correct)

Use the loss formula:

$$
\text{Loss} = -\sum y_{\text{true}} \cdot \log(y_{\text{pred}})
= -\log(0.659) ≈ 0.417
$$

---

### **4. Backward Pass — Gradient of Loss**

The gradient when using softmax + cross-entropy simplifies to:

$$
\text{grad} = y_{\text{pred}} - y_{\text{true}} = [0.098, 0.659, 0.242] - [0, 1, 0] = [0.098, -0.341, 0.242]
$$

---

### **5. Compute Weight Gradient**

We now compute how to change the weights using:

$$
\text{W update} = X^\top \cdot \text{grad}
$$

Where:

* `X` is shape (1, 2)
* `grad` is shape (1, 3)
* So `X.T` is (2, 1)

$$
\text{W update} = 
\begin{bmatrix} 1.0 \\ 2.0 \end{bmatrix} 
\cdot 
\begin{bmatrix} 0.098 & -0.341 & 0.242 \end{bmatrix}
=
\begin{bmatrix}
1.0 * 0.098 & 1.0 * -0.341 & 1.0 * 0.242 \\
2.0 * 0.098 & 2.0 * -0.341 & 2.0 * 0.242
\end{bmatrix}
=
\begin{bmatrix}
0.098 & -0.341 & 0.242 \\
0.196 & -0.682 & 0.484
\end{bmatrix}
$$

If learning rate = 0.1:

$$
\Delta W = -0.1 \cdot \text{W update} = 
\begin{bmatrix}
-0.0098 & 0.0341 & -0.0242 \\
-0.0196 & 0.0682 & -0.0484
\end{bmatrix}
$$

This tells us how to **adjust weights** to reduce the loss.

---

### ✅ Final Summary

| Step | What Happens                     | Result                                  |
| ---- | -------------------------------- | --------------------------------------- |
| 1    | Multiply input by weights        | `logits = [-0.8, 1.1, 0.1]`             |
| 2    | Apply softmax                    | `y_pred = [0.098, 0.659, 0.242]`        |
| 3    | Compute cross-entropy loss       | `loss ≈ 0.417`                          |
| 4    | Get gradient (y\_pred - y\_true) | `[0.098, -0.341, 0.242]`                |
| 5    | Compute weight update            | Use `X.T * grad` to get gradient matrix |

---

