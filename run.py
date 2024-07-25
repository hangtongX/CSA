import os.path
from pathlib import Path
import munch
import toml
import torch, gc

from utils.logger_handler import get_logger
from utils.getters import get_trainer
from trainer.train import BaseTrainer

if __name__ == '__main__':
    file_path = Path(os.path.abspath(__file__)).resolve()
    config = munch.munchify(toml.load(os.path.join(file_path.parents[-4], 'run.toml')))
    logger = get_logger(
        file_path=os.path.join(
            file_path.parents[-4], 'run_error_log-' + str(config.train.device) + '.log'
        )
    )

    for dataname in config.projects.datasets:
        for modelname in config.projects.models:
            try:
                try:
                    if config.projects.trainer == '':
                        trainer_name = None
                    else:
                        trainer_name = config.projects.trainer
                    back = get_trainer(model_name=modelname, trainer_name=trainer_name)
                    logger.info(back.info)
                    trainer = back.trainer(modelname, dataname, config=config)
                except Exception as e:
                    logger.info(e)
                    trainer = BaseTrainer(modelname, dataname, config=config)

                # begin train
                result = trainer.train_test_for_times(config.run.times)
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                logger.exception(e)
