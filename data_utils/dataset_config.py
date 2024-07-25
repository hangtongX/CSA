from model.base.baseconfig import BaseConfig
from typing import Literal
from dataclasses import dataclass

@dataclass
class DatasetConfig(BaseConfig):

    label: bool = None
    negative_num: int = 1
    concact: bool = True
    testSize: int = None
    trainSize: int = None
    valSize: int = None
    dataname: str = None
    path: str = None
    compression: Literal['gzip'] = None
    split:bool = False
    split_type = None
    traintype: Literal['normal', 'rec', 'rec_neg', 'rec_pairneg'] = None
    evaltype: Literal['normal', 'rec', 'rec_neg', 'rec_pairneg'] = None
    testtype: Literal['normal', 'rec', 'rec_neg', 'rec_pairneg'] = None
