"""
Implementation of a PINN to solve the two dimensional heat equation.
Configurations are stored in the config.py file.

Of course that here you have to change your initial condition, boundary conditions, etc.
And the collocation points, etc.
"""
import logging
import torch
import torch.nn as nn
from config import Network, PINN, Plot
from utils import compute_residual, initial_condition
from utils import plots


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
torch.manual_seed(123)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layer = [nn.Linear(Network().neuron_inputs,
                           Network().neuron_hidden), nn.GELU()]
        for i in range(Network().hidden_layers_numbers):
            layer += [nn.Linear(Network().neuron_hidden,
                                Network().neuron_hidden), nn.GELU()]
        layer += [nn.Linear(Network().neuron_hidden,
                            Network().neuron_outputs), nn.GELU()]
        self.net = nn.Sequential(*layer)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


def train_pinn():
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=Network().lr)

    collocation_residual = PINN.num_collocation_res
    collocation_ic = PINN.num_collocation_ic
    collocation_bc = PINN.num_collocation_bc
    lambda_residual = PINN.lambda_residual
    lambda_ic = PINN.lambda_ic
    lambda_bc = PINN.lambda_bc

    # Residual Collocation
    x_col_res = torch.rand(collocation_residual, 1)
    t_col_res = torch.rand(collocation_residual, 1)

    # Initial Condition Collocation
    x_col_ic = torch.rand(collocation_ic, 1)
    t_col_ic = torch.zeros((collocation_ic, 1))

    # Boundary Condition Collocation
    t_x_bc = torch.rand(collocation_bc, 1)
    x_bc = torch.zeros((collocation_bc, 1), requires_grad=True)
    t_l_bc = torch.rand(collocation_bc, 1)
    l_bc = torch.ones((collocation_bc, 1), requires_grad=True)

    # Neumann
    ux_0_bc = torch.zeros((collocation_bc, 1))
    ux_1_bc = torch.zeros((collocation_bc, 1))

    # Snapshot values
    snapshots = torch.zeros((Plot().snap_x,
                             Plot().snap_t,
                             Plot().frames_snap))

    logger.info(f"Training PINN for {Network().epochs} epochs")
    for _ in range(Network().epochs):
        optimizer.zero_grad()
        if _ % 100 == 0:
            logger.info(f"Epoch {_} of {Network().epochs}")
        # Residual
        residual = compute_residual(model, x_col_res, t_col_res)
        loss_residual = torch.mean(residual**2)

        # Initial
        model_ic = model(x_col_ic, t_col_ic)
        loss_ic = torch.mean((model_ic-initial_condition(x_col_ic))**2)

        # Boundary
        u_0_bc = model(x_bc, t_x_bc)
        du_0_bc = torch.autograd.grad(
            u_0_bc, x_bc, grad_outputs=torch.ones_like(u_0_bc),
            create_graph=True
        )[0]

        u_l_bc = model(l_bc, t_l_bc)
        du_l_bc = torch.autograd.grad(
            u_l_bc, l_bc, grad_outputs=torch.ones_like(u_l_bc),
            create_graph=True
        )[0]

        loss_0_bc = torch.mean((du_0_bc-ux_0_bc)**2)
        loss_1_bc = torch.mean((du_l_bc-ux_1_bc)**2)
        loss_b = (loss_0_bc + loss_1_bc)
        loss = lambda_residual*loss_residual+lambda_ic*loss_ic+lambda_bc*loss_b
        loss.backward()
        optimizer.step()

    torch.save(
            {'model_state_dict': model.state_dict()}, Network().save_path
        )
    logger.info(f"PINN trained for {Network().epochs} epochs")
    return model, snapshots


def main(flag: bool = False):
    if flag:
        model = NeuralNetwork()
        loaded = torch.load(Network().save_path)
        model.load_state_dict(loaded["model_state_dict"])
        model.eval()
        plotter = plots()
        plotter.heat_comparation(model)
    else:
        model = train_pinn()


if __name__ == "__main__":
    main()
