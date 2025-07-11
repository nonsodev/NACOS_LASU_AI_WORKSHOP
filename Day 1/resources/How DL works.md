# How DL works

---

### 🧮 1. Starting With a Table (Dataset)

| ID | Hours Studied (x) | Test Score (y) |
| --- | --- | --- |
| 1 | 1 | 50 |
| 2 | 2 | 55 |
| 3 | 3 | 65 |
| 4 | 4 | 70 |

We want a model to **predict `y` from `x`**, i.e., predict test scores based on hours studied.

---

### 🧠 2. Define a Simple Neural Network

Let's use a **1-layer neural net (linear regression)**:

$$
\hat{y} = wx + b
$$

Where:

- `w` = weight (how much each hour affects the score)
- `b` = bias (base score even with 0 study)

Initial values:

- w = 10, b = 40

---

### 🔢 3. Forward Pass — Make Predictions

Use model to predict \hat{y} for each input `x`:

| x | y (true) | 𝑤𝑥 + 𝑏 = 𝑦̂ (predicted) |
| --- | --- | --- |
| 1 | 50 | 10×1 + 40 = 50 |
| 2 | 55 | 10×2 + 40 = 60 |
| 3 | 65 | 10×3 + 40 = 70 |
| 4 | 70 | 10×4 + 40 = 80 |

---

### 📉 4. Loss Function (Error)

Use **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{N} \sum (\hat{y} - y)^2
$$

Calculate:

$$
\text{MSE} = \frac{1}{4} [(50-50)^2 + (60-55)^2 + (70-65)^2 + (80-70)^2] = \frac{0 + 25 + 25 + 100}{4} = 37.5
$$

---

### 🔁 5. Backpropagation — Compute Gradients

We want to adjust `w` and `b` to **minimize loss**.

Gradients tell us **which direction and how much to change weights**:

$$
\frac{\partial L}{\partial w} = \frac{2}{N} \sum (wx + b - y) \cdot x
$$

$$
\frac{\partial L}{\partial b} = \frac{2}{N} \sum (wx + b - y)
$$

Use these formulas to compute how the weight and bias should change.

---

### 🔧 6. Gradient Descent — Update Parameters

New weights:

$$
w = w - \alpha \cdot \frac{\partial L}{\partial w}  
\quad\text{and}\quad
b = b - \alpha \cdot \frac{\partial L}{\partial b}
$$

Where $\alpha$ is the **learning rate** (e.g., 0.01).

We repeat steps:

- forward pass → loss → backprop → update
- This is called an **epoch**.

---

### 🔄 7. Repeat Until Loss is Low

Over many epochs, the model learns better `w` and `b` values:

Eventually:

$$
\hat{y} \approx y
$$

And the model can predict accurately.

---

### Summary Table:

| Step | What Happens |
| --- | --- |
| Dataset | Tabular input/output (x, y) |
| Model | $\hat{y} = wx + b$ |
| Forward Pass | Compute predictions \hat{y} |
| Loss | MSE compares $\hat{y}$ vs. y |
| Backpropagation | Compute gradient of loss w.r.t.`w`,`b` |
| Gradient Descent | Update`w`,`b`using gradients |
| Loop | Repeat to improve prediction |

---