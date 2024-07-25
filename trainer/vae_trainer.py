import logging
from copy import deepcopy
import torch.distributed as dist
import munch
import pandas as pd
import torch
from colorama import Fore, init as colorama_init
from model.base.basemodel import BaseModel
from trainer.train import BaseTrainer

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)
colorama_init(autoreset=True)


class Trainer(BaseTrainer):
    def __init__(
        self,
        modelname: str,
        dataname: str,
        config: munch.Munch = None,
    ):
        super(Trainer, self).__init__(modelname, dataname, config)

    def _setup_model(self):
        history_data = self.train_dataset.histiory_items()
        self.model.model_config.update(self.dataInfo)
        self.model.device = self.device
        self.model.build(history_data=history_data)
        self.model.to(self.device)
