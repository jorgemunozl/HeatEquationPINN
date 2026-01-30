from dataclasses import dataclass
import os


@dataclass
class Network:
    # Network configuration
    neuron_inputs: int = 2
    neuron_hidden: int = 100
    hidden_layers_numbers: int = 4
    neuron_outputs: int = 1

    # Training configuration
    epochs: int = 7000
    lr: float = 1e-3

    # Saving configuration
    save_dir: str = "parameters"
    save_name: str = "parameters.pth"
    save_path: str = os.path.join(save_dir, save_name)


@dataclass
class Plot:
    sample: int = 100
    snapshot_step: int = 10
    snap_x: int = 1000
    snap_t: int = 100
    frames_snap: int = 100


@dataclass
class PINN:
    # Heat equation alpha parameter
    alpha: float = 0.1

    # Collocation points configuration
    num_collocation_res: int = 1000
    num_collocation_ic: int = 500
    num_collocation_bc: int = 600

    # Lambda weights for the loss function
    lambda_residual: float = 10.0
    lambda_ic: float = 6.0
    lambda_bc: float = 5.0

    # Error configuration
    error_x_sample: int = 1000
    error_t_sample: int = 100
