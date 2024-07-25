from model.base.baseconfig import BaseRecConfig
from dataclasses import dataclass


@dataclass
class Config(BaseRecConfig):

    name: str = None
    layers: list = None
    drop_out: int = None
    lat_dim: int = None
    anneal_cap: float = None
    total_anneal_steps: int = None
    encoder_type: str = None
    decoder_type: str = None
    concepts_k: int = None
    tau: float = None
    mask: bool = None
    mask_all: bool = None
    mask_local: bool = None
    masked_graph: list = None
    alpha: float = None
    l_a: float = None
    rol: float = None
