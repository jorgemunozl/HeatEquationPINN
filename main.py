import torch
import torch.nn as nn
from config import netConfig, pinnConfig, plotConfig
from utils import compute_residual, initial_condition
from utils import plots


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        layer = [nn.Linear(netConfig().neuron_inputs,
                           netConfig().neuron_hidden), nn.Tanh()]
        for i in range(netConfig().hidden_layers_numbers):
            layer += [nn.Linear(netConfig().neuron_hidden,
                                netConfig().neuron_hidden), nn.Tanh()]
        layer += [nn.Linear(netConfig().neuron_hidden,
                            netConfig().neuron_outputs), nn.Tanh()]
        self.net = nn.Sequential(*layer)

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


def train_pinn():

    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=netConfig().lr)
    num_collocation_res = pinnConfig().num_collocation_res
    num_collocation_ic = pinnConfig().num_collocation_ic
    num_collocation_bc = pinnConfig().num_collocation_bc
    lambda_residual = pinnConfig().num_collocation_bc
    lambda_ic = pinnConfig().num_collocation_bc
    lambda_bc = pinnConfig().lambda_bc

    # Residual Collocation
    x_col_res = torch.rand(num_collocation_res, 1)
    t_col_res = torch.rand(num_collocation_res, 1)

    # Initial Condition Collocation
    x_col_ic = torch.rand(num_collocation_ic, 1)
    t_col_ic = torch.zeros((num_collocation_ic, 1))

    # Boundary Condition Collocation
    t_x_bc = torch.rand(num_collocation_bc, 1)
    x_bc = torch.zeros((num_collocation_bc, 1), requires_grad=True)
    t_l_bc = torch.rand(num_collocation_bc, 1)
    l_bc = torch.ones((num_collocation_bc, 1), requires_grad=True)

    # Neumann
    ux_0_bc = torch.zeros((num_collocation_bc, 1))
    ux_1_bc = torch.zeros((num_collocation_bc, 1))

    # Snapshot values

    snapshots = torch.zeros((plotConfig().snap_x,
                             plotConfig().snap_t,
                             plotConfig().frames_snap))

    for _ in range(netConfig().epochs):
        optimizer.zero_grad()

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
        plotter = plots()
        if _ % plotConfig().snapshot_step == 0:
            if _ == netConfig().epochs-1:
                plotter.animate_snapshot(model, snapshots, _, True)
            else:
                plotter.animate_snapshot(model, snapshots, _, False)

    save_path = netConfig().save_path
    torch.save(
            {'model_state_dict': model.state_dict()}, save_path
        )
    return model, snapshots


if __name__ == "__main__":
    """
    model = NeuralNetwork()
    loaded = torch.load(netConfig().save_path)
    model.load_state_dict(loaded["model_state_dict"])
    model.eval()
    plotter = plots()
    plotter.animation_mape(model)
    """
    model = train_pinn()
