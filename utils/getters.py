import os
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Type

from colorama import Fore, init as colorama_init
import munch
from pathlib import Path
from data_utils.dataset import BaseDataset
from model.base.baseconfig import BaseConfig, BaseRecConfig
from data_utils.dataset import generate_dataset

colorama_init(autoreset=True)
file_path = Path(os.path.abspath(__file__)).resolve()
parent_dir = file_path.parents[-4]


def get_model(model_name: str):
    """
    Given the model name , back the model class
    @param model_name:
    @return:
    """
    model_file_name = model_name.upper()
    model_path = '.'.join(['model', model_file_name.lower(), 'model'])
    if find_spec(model_path):
        return getattr(import_module(model_path), model_file_name)
    else:
        raise Exception(f'{model_file_name} is not exist in {model_path}, please check')


def get_model_params_class(model_name: str):
    model_file_name = 'Config'
    model_path = '.'.join(['model', model_name.lower(), 'config'])
    if find_spec(model_path):
        return getattr(import_module(model_path), model_file_name)
    else:
        raise Exception(f'{model_file_name} is not exist in {model_path}, please check')


def get_model_params_rec(model_name: str) -> BaseConfig:
    """
    Given the model name , back the model parameters
    @param model_name:
    @return:
    """
    file_name = model_name.upper()
    param_path = os.path.join(parent_dir, 'params', file_name + '.toml')
    if os.path.exists(param_path):
        config = get_model_params_class(model_name)()
        return config.from_toml_file(param_path)
    else:
        raise FileNotFoundError(
            f'The parameters of {file_name} is not exist in {param_path}, please check'
        )


def get_lossF(lossname: str):
    loss_name = lossname.name.upper()
    path = '.'.join(['loss', 'loss'])

    if find_spec(path, __name__):
        return getattr(import_module(path, __name__), loss_name)
    else:
        raise Exception(f'{path} is not exist in {loss_name}, please check')


def get_dataset(dataset_name: str, config: munch.Munch) -> BaseDataset:
    dataconfig = config.dataset
    dataconfig.dataname = dataset_name
    return generate_dataset(dataconfig)


def get_loss(lossname: str):
    lossname = lossname.upper()
    path = '.'.join(['loss', 'cal_loss'])
    if find_spec(path, __name__):
        return getattr(import_module(path, __name__), lossname)
    else:
        raise Exception(Fore.RED + f'{path} is not exist in {lossname}, please check')


def get_trainer(model_name: str, trainer_name: str = None):
    model_file_name = 'Trainer'
    model_path = '.'.join(['model', model_name.lower(), 'train'])
    if trainer_name is not None:
        trainer_path = '.'.join(['trainer', trainer_name.lower()])
    if find_spec(model_path):
        trainer = getattr(import_module(model_path), model_file_name)
        info = Fore.YELLOW + f'The model:{model_name} have a unique trainer'
    elif trainer_name is not None:
        trainer = getattr(import_module(trainer_path), model_file_name)
        info = (
            Fore.YELLOW
            + f'The model {model_name} use the unique trainer {trainer_name} by setting'
        )
    else:
        raise Warning(
            f'The model:{model_name} does not have a unique trainer and will use Basetrainer'
        )

    return munch.Munch({'trainer': trainer, 'info': info})
