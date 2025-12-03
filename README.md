# Inference for Censored Data with DNN

The simulations includes the finite sample performance of the proposed estimation procedure, the one-sample test, the two-sample test, and the goodness-of-fit test. We conducted three real-world data analyses. The first  is about the Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatment (SUPPORT), which is publicly available at https://biostat.app.vumc.org/wiki/Main/DataSets. The second drew on data from TCGA, accessible via https://github.com/PathologyDataScience/SCNN. the Breast Cancer Gene Expression Profiles
(METABRIC) dataset, available at https://www.kaggle.com/datasets/raghadalharbi/breastcancer-gene-expression-profiles-metabric.

## Reference to download and install (Python 3.12)

+ pip install packages

> pip install numpy pandas torch matplotlib scipy pycox lifelines


## Usage of DNN Training 
> We use **PyTorch**.

```
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(n_input, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, n_output))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)
    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    g_train = model(X_train)
```

## Gaussian Quadrature

Gaussian quadrature is a numerical integration method used to compute definite integrals with high accuracy. It transforms the integral into a weighted sum by selecting appropriate integration points and weights.

---

### Basic Principle

The goal of Gaussian quadrature is to approximate the definite integral by carefully choosing specific points (called **Gaussian nodes**) in the integration interval and associated weights. The formula is:

$$
\int_a^b f(x) dx \approx \sum_{i=1}^n w_i f(x_i)
$$

Where:
- \(x_i\) are the **Gaussian nodes** within the integration interval.
- \(w_i\) are the weights corresponding to each node.
- \(n\) is the number of integration points (i.e., the number of Gaussian nodes).

One major advantage of Gaussian quadrature is that if the integrand \(f(x)\) is a polynomial \(P(x)\) of degree less than \(2n\), the integral is computed **exactly**.

---

### Steps of Gaussian Quadrature

#### 1. Standard Interval Quadrature
Gaussian quadrature is typically applied on the standard interval \([-1, 1]\). The formula for this case is:

$$
\int_{-1}^1 f(x) dx \approx \sum_{i=1}^n w_i f(x_i)
$$

#### 2. Transforming to a General Interval
For an arbitrary interval \([a, b]\), a variable substitution is performed to map the interval to \([-1, 1]\):

$$
x = \frac{b-a}{2} \xi + \frac{b+a}{2}, \quad dx = \frac{b-a}{2} d\xi
$$

The integral becomes:

$$
\int_a^b f(x) dx = \frac{b-a}{2} \int_{-1}^1 f\left(\frac{b-a}{2} \xi + \frac{b+a}{2}\right) d\xi \approx \frac{b-a}{2} \sum_{i=1}^n w_i f\left(\frac{b-a}{2} \xi_i + \frac{b+a}{2}\right)
$$

Then Gaussian quadrature can be applied on the transformed integral.

---
### Gaussian Nodes and Weights

The Gaussian nodes \(x_i\) are the roots of orthogonal polynomials (e.g., Legendre polynomials) on the interval \([-1, 1]\), while the weights \(w_i\) are computed as:

$$
w_i = \int_{-1}^1 \prod_{j \neq i} \frac{\xi - x_j}{x_i - x_j} d\xi,
$$

For various values of \(n\), the nodes and weights can be computed numerically or obtained from precomputed tables.

---

### Example 1

Below is an example implementation of Gaussian quadrature in Python:

```python
import numpy as np

def gaussian_quadrature(func, a, b, n):
    # Get Gaussian nodes and weights
    x, w = np.polynomial.legendre.leggauss(n)
    
    # Transform interval [a, b] to [-1, 1]
    t = 0.5 * (x + 1) * (b - a) + a
    transformed_weights = 0.5 * (b - a) * w
    
    # Compute the weighted sum
    integral = np.sum(transformed_weights * func(t))
    return integral

result = gaussian_quadrature(lambda x: x**2, 0, 1, 3)
print(f"Integral result: {result}")

```




---
## Adaptive Gaussian Quadrature

#### 1. Core Idea

Automatically subdivide the integration interval where the integrand changes rapidly, so we spend more samples only where needed to achieve a target accuracy.


#### 2. Key Mechanism

On a subinterval $[a, b]$, compute two Gaussian quadrature estimates: a higher-order estimate and a lower-order estimate.
Use their difference as a local error estimate.
If the error is within tolerance, accept the higher-order result; otherwise, split $[a, b]$ into two halves and recurse.


#### 3. Stopping Criteria

Local error $\le$ tolerance (absolute and/or relative).
Reaching a maximum recursion depth or a minimum interval size (to avoid infinite subdivision).


### Example 2

Below is an example implementation of adaptive Gaussian quadrature in Python:

```python


import numpy as np

def quad_once(func, a, b, nodes_low, weights_low, nodes_high, weights_high):
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    t_low = mid + half * nodes_low
    t_high = mid + half * nodes_high
    f_low = func(t_low)
    f_high = func(t_high)
    I_low = half * np.sum(weights_low * f_low)
    I_high = half * np.sum(weights_high * f_high)
    return I_low, I_high, abs(I_high - I_low)

def adaptive_gaussian_quadrature(func, a, b, n_low=32, n_high=64, tol=1e-5, max_depth=20):
    x_low, w_low = np.polynomial.legendre.leggauss(n_low)
    x_high, w_high = np.polynomial.legendre.leggauss(n_high)

    total = 0.0
    stack = [(a, b, 0)]  # (left, right, depth)

    while stack:
        left, right, depth = stack.pop()
        I_low, I_high, err = quad_once(func, left, right, x_low, w_low, x_high, w_high)

        if err <= tol or depth >= max_depth:
            total += I_high
        else:
            mid = 0.5 * (left + right)
            stack.append((mid, right, depth + 1))
            stack.append((left, mid, depth + 1))

    return total

f = lambda x: x**2
res = adaptive_gaussian_quadrature(f, 0.0, 1.0, n_low=32, n_high=64, tol=1e-10, max_depth=20)
print("Integral result:", res)

```




