from pydantic import BaseModel, Field
from typing import Annotated


class netConfig(BaseModel):
    save_path: str = Field(
        default="exp/parameters.pth",
        description="Parameter's path"
    )
    neuron_inputs: int = Field(
        default=1,
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
        default=2
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


class pinnConfig(BaseModel):
    alpha: Annotated[int, {}] = Field(
        default=0.1,
        description=""
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
