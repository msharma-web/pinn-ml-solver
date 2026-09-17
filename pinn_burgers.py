import torch
import torch.nn as nn
import numpy as np

# 1. Physics-Informed Neural Network Architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        # Concatenate inputs (x, t)
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

def physics_loss(model, x, t, nu=0.01/np.pi):
    """Computes the PDE residual loss using Automatic Differentiation"""
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    # First derivatives
    u_g = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    
    # Second derivative w.r.t x
    u_xx = torch.autograd.grad(u_g, x, torch.ones_like(u_g), create_graph=True)[0]
    
    # Burgers' Equation residual: u_t + u*u_x - nu*u_xx = 0
    f = u_t + u * u_g - nu * u_xx
    return torch.mean(f**2)

# 2. Synthetic Boundary & Initial Condition Data Generator
def get_training_data(n_bc=50, n_pde=1000):
    # Boundary / Initial conditions
    t_zero = torch.zeros((n_bc, 1))
    x_init = torch.linspace(-1, 1, n_bc).view(-1, 1)
    u_init = -torch.sin(np.pi * x_init) # u(x,0) = -sin(pi*x)
    
    # Collocation points inside the space-time domain x in [-1,1], t in [0,1]
    x_pde = (2 * torch.rand((n_pde, 1)) - 1)
    t_pde = torch.rand((n_pde, 1))
    
    return x_init, t_zero, u_init, x_pde, t_pde

if __name__ == "__main__":
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_init, t_zero, u_init, x_pde, t_pde = get_training_data()

    print("Training Physics-Informed Neural Network (PINN)...")
    for epoch in range(1001):
        optimizer.zero_grad()
        
        # Loss 1: Initial Condition Loss
        u_pred_init = model(x_init, t_zero)
        loss_ic = torch.mean((u_pred_init - u_init)**2)
        
        # Loss 2: Physics (PDE Residual) Loss
        loss_pde = physics_loss(model, x_pde, t_pde)
        
        # Total Loss
        total_loss = loss_ic + loss_pde
        total_loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}")
