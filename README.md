# Physics-Informed Neural Network (PINN) for Non-Linear PDEs

A PyTorch implementation of a Physics-Informed Neural Network (PINN) designed to solve the continuous-time **1D Burgers' Equation** ($\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$) without requiring labelled simulation grids.

---

## **Key Highlights**
* **Physics-Embedded Loss Function:** Embedded physical laws directly into the neural network optimization objective via automatic differentiation (`torch.autograd`).
* **Data-Efficient Learning:** Approximates PDE solutions across space-time coordinates $(x, t)$ using minimal boundary initial points and collocation point sampling.
* **Framework:** PyTorch, NumPy, Matplotlib.

---

## **Repository Architecture**

```text
├── pinn_burgers.py    # Main training loop, network architecture, and PDE loss computation
├── README.md          # Project documentation
└── requirements.txt   # Dependencies
