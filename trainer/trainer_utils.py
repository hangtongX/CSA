import logging
import random
from colorama import Fore, init as color_init
import numpy as np
import torch
from tqdm import tqdm

from model.base.baseconfig import BaseConfig
from trainer.trainConfig import BaseTrainerConfig
from importlib.util import find_spec

logger = logging.getLogger(__name__)
color_init(autoreset=True)


def set_seed(seed: int):
    """
    Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``

    Args:

        seed (int): The seed to be applied
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wandb_is_available():
    return find_spec("wandb") is not None


def rename_logs(logs):
    train_prefix = "train_"
    eval_prefix = "eval_"

    clean_logs = {}

    for metric_name in logs.keys():
        if metric_name.startswith(train_prefix):
            clean_logs[metric_name.replace(train_prefix, "train/")] = logs[metric_name]

        if metric_name.startswith(eval_prefix):
            clean_logs[metric_name.replace(eval_prefix, "eval/")] = logs[metric_name]

    return clean_logs


class TrainingCallback:
    """
    Base class for creating training callbacks
    """

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of training.
        """

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of training.
        """

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of an epoch.
        """

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of an epoch.
        """

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a training step.
        """

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a training step.
        """

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """

    def on_test_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """

    def on_test_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called at the end of a evaluation step.
        """

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after an evaluation phase.
        """

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a prediction phase.
        """

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        """
        Event called after logging the last logs.
        """

    def __repr__(self) -> str:
        return self.__class__.__name__


class WandbCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `wandb` (https://wandb.ai/).

    It allows users to store their configs, monitor their trainings
    and compare runs through a graphic interface. To be able use this feature you will need:

        - a valid `wandb` account
        - the package `wandb` installed in your virtual env. If not you can install it with

        .. code-block::

            $ pip install wandb

        - to be logged in to your wandb account using

        .. code-block::

            $ wandb login
    """

    def __init__(self, model_name: str = None):
        if not wandb_is_available():
            raise ModuleNotFoundError(
                "`wandb` package must be installed. Run `pip install wandb`"
            )

        else:
            import wandb

            self._wandb = wandb
        self._wandb.require('core')
        self.is_initialized = False
        self.model_name = model_name

    def setup(
        self,
        training_config: BaseTrainerConfig,
        model_config: BaseConfig = None,
        **kwargs,
    ):
        """
        Setup the WandbCallback.

        args:
            training_config (BaseTrainerConfig): The training configuration used in the run.

            model_config (BaseAEConfig): The model configuration used in the run.

            project_name (str): The name of the wandb project to use.

            entity_name (str): The name of the wandb entity to use.
        """

        self.is_initialized = True

        training_config_dict = training_config.__dict__

        if training_config.run_name == '':
            if model_config.loss_func is None:
                run_name = '-'.join([model_config.name[0], self.model_name])
            else:
                run_name = '-'.join(
                    [model_config.name[0], self.model_name, model_config.loss_func]
                )
        else:
            run_name = training_config.run_name
        self.run = self._wandb.init(
            project=training_config.project_name,
            entity=training_config.entity_name,
            name=run_name,
            tags=training_config.run_tags,
        )

        if model_config is not None:
            model_config_dict = model_config.to_dict()

            self._wandb.config.update(
                {
                    "training_config": training_config_dict,
                    "model_config": model_config_dict,
                }
            )

        else:
            self._wandb.config.update({**training_config_dict})

        self._wandb.define_metric('train/step')
        self._wandb.define_metric('train/*', step_metric='train/step')
        self._wandb.define_metric('eval/*', step_metric='train/step')

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        model_config = kwargs.pop("model_config", None)
        if not self.is_initialized:
            self.setup(training_config, model_config=model_config)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        step = kwargs.pop("step", None)
        logs = rename_logs(logs)

        self._wandb.log({**logs, 'train/step': step})

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        rtype = kwargs.pop('type', 'table')
        name = kwargs.pop("name", 'predict')
        if rtype == 'table':
            table = self._wandb.Table(dataframe=kwargs['table'])
            self._wandb.log({name: table})
        else:
            raise NotImplementedError(
                Fore.RED
                + "Unrecognized data type in prediction step when save to wandb."
            )

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.run.finish()


class CallbackHandler:
    """
    Class to handle list of Callback.
    """

    def __init__(self, callbacks, model):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks but there one is already used."
                f" The current list of callbacks is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_init_end", training_config, **kwargs)

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_begin", training_config, **kwargs)

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_step_end", training_config, **kwargs)

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_begin", training_config, **kwargs)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_eval_step_end", training_config, **kwargs)

    def on_test_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_test_step_begin", training_config, **kwargs)

    def on_test_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_test_step_end", training_config, **kwargs)

    def on_train_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_begin", training_config, **kwargs)

    def on_train_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_train_end", training_config, **kwargs)

    def on_epoch_begin(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_begin", training_config, **kwargs)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_epoch_end", training_config)

    def on_evaluate(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_evaluate", **kwargs)

    def on_save(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_save", training_config, **kwargs)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        self.call_event("on_log", training_config, logs=logs, **kwargs)

    def on_prediction_step(self, training_config: BaseTrainerConfig, **kwargs):
        self.call_event("on_prediction_step", training_config, **kwargs)

    def call_event(self, event, training_config, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                training_config,
                model=self.model,
                **kwargs,
            )


class MetricConsolePrinterCallback(TrainingCallback):
    """
    A :class:`TrainingCallback` printing the training logs in the console.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # make it print to the console.
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.INFO)

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        logger = kwargs.pop("logger", self.logger)
        times = kwargs.pop("times", 1)
        rank = kwargs.pop("rank", -1)

        if logger is not None and (rank == -1 or rank == 0):
            epoch_train_loss = logs.get(f"train_{times}", None)
            epoch_eval_loss = logs.get(f"eval_{times}", None)

            # logger.info(
            #     "--------------------------------------------------------------------------"
            # )
            # if epoch_train_loss is not None:
            #     logger.info(f"Train loss: {np.round(epoch_train_loss, 4)}")
            # if epoch_eval_loss is not None:
            #     logger.info(f"Eval loss: {np.round(epoch_eval_loss, 4)}")
            # logger.info(
            #     "--------------------------------------------------------------------------"
            # )


class ProgressBarCallback(TrainingCallback):
    """
    A :class:`TrainingCallback` printing the training progress bar.
    """

    def __init__(self):
        self.train_progress_bar = None
        self.eval_progress_bar = None
        self.test_progress_bar = None
        self.train_bar = None

    def on_train_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        train_loader = kwargs["train_loader"]
        rank = kwargs.pop("rank", -1)
        loss = kwargs['loss']
        if train_loader is not None:
            if rank == 0 or rank == -1:
                self.train_progress_bar = tqdm(
                    total=len(train_loader),
                    unit="batch",
                    desc=Fore.GREEN
                    + f"Training of epoch {epoch}/{training_config.num_epochs}",
                    position=1,
                    ncols=100,
                    leave=False,
                    colour='green',
                )

    def on_eval_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        eval_loader = kwargs["eval_loader"]
        rank = kwargs.pop("rank", -1)
        if eval_loader is not None:
            if rank == 0 or rank == -1:
                self.eval_progress_bar = tqdm(
                    total=len(eval_loader),
                    unit="batch",
                    desc=Fore.LIGHTBLUE_EX
                    + f"Eval of epoch {epoch}/{training_config.num_epochs}",
                    ncols=100,
                    position=2,
                    leave=False,
                    colour='yellow',
                )

    def on_train_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.train_progress_bar is not None:
            loss = kwargs['loss']
            self.train_progress_bar.set_postfix(loss=round(loss, 4))
            self.train_progress_bar.update(1)

    def on_eval_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.eval_progress_bar is not None:
            self.eval_progress_bar.update(1)

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwags):
        if self.train_progress_bar is not None:
            self.train_progress_bar.close()

        if self.eval_progress_bar is not None:
            self.eval_progress_bar.close()

    def on_test_step_begin(self, training_config: BaseTrainerConfig, **kwargs):
        epoch = kwargs.pop("epoch", None)
        test_loader = kwargs["test_loader"]
        rank = kwargs.pop("rank", -1)
        if test_loader is not None:
            if rank == 0 or rank == -1:
                self.test_progress_bar = tqdm(
                    total=len(test_loader),
                    unit="batch",
                    desc=Fore.LIGHTBLUE_EX
                    + f"Test of epoch {epoch}/{training_config.num_epochs}",
                    position=3,
                    ncols=100,
                    leave=False,
                    colour='red',
                )

    def on_test_step_end(self, training_config: BaseTrainerConfig, **kwargs):
        if self.test_progress_bar is not None:
            self.test_progress_bar.update(1)
