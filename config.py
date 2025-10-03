from pydantic import BaseModel, Field


class netConfig(BaseModel):
    save_path: str = Field(
        default="parameters_snap.pth",
        description="Parameter's path"
    )
    neuron_inputs: int = Field(
        default=2,
        description='Number of neurons'
    )
    neuron_hidden: int = Field(
        default=100,
        description='Number of neurons'
    )
    hidden_layers_numbers: int = Field(
        default=8,
    )
    neuron_outputs: int = Field(
        default=1
    )
    epochs: int = Field(
        default=1000,
        description='Number of times that the parameter actualize'
    )
    lr: float = Field(
        default=1e-3,
        description='Learning rate'
    )


class plotConfig(BaseModel):
    sample: int = Field(
        default=100,
        description='Number o'
    )
    snapshot_step: int = Field(
        default=10,
        description=""
    )
    snap_x: int = Field(
        default=1000,
        description=""
    )
    snap_t: int = Field(
        default=100,
        description=""
    )
    frames_snap: int = Field(
        default=100,
        description=""
    )


class pinnConfig(BaseModel):
    alpha: float = Field(
        default=0.1,
        description="Important for the PDE"
    )

    num_collocation_res: int = Field(
        default=500,
        description=""
    )
    num_collocation_ic: int = Field(
        default=100,
        description=''
    )
    num_collocation_bc: int = Field(
        default=200,
        description=''
    )
    lambda_residual: float = Field(
        default=10.0,
        description=''
    )
    lambda_ic: float = Field(
        default=10.0,
        description=''
    )
    lambda_bc: float = Field(
        default=10.0,
        description=''
    )
    error_x_sample: int = Field(
        default=10000,
        description=''
    )
    error_t_sample: int = Field(
        default=100,
        description=''
    )
