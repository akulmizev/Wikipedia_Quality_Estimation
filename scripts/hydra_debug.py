import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from wqe.experiment.experiment import ExperimentRunner
from wqe.utils.config import MainConfig

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)


@hydra.main(version_base=None, config_path="../config/wqe", config_name="config")
def run_experiment(config_dict: DictConfig) -> None:

    config = MainConfig(**config_dict)

    runner = ExperimentRunner(config)
    runner.run_experiment()


if __name__ == "__main__":
    run_experiment()
