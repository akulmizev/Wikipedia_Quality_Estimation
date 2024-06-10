import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from ..experiment.experiment import ExperimentRunner
from ..utils.config import MainConfig

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=None, config_name="config")
def run_experiment(config_dict: DictConfig) -> None:

    try:
        config_dict = OmegaConf.to_container(config_dict, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue:
        logger.error("Experiment config must be provided, e.g. `+experiment=basic`.")
        exit(1)

    config = MainConfig(**config_dict)

    runner = ExperimentRunner(config)
    runner.run_experiment()


if __name__ == "__main__":
    run_experiment()
