import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.wqe.experiment.experiment import ExperimentRunner
from src.wqe.utils.config import *

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)

@hydra.main(version_base=None, config_path="config/hydra_debug", config_name="config")
def run_experiment(config_dict: DictConfig) -> None:

    config = MainConfig(**config_dict)

    runner = ExperimentRunner(config)
    runner.run_experiment()


if __name__ == "__main__":
    run_experiment()
