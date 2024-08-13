import dataclasses
import logging

import submitit
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

    # NOTE: there is a submitit plugin for hydra, but it seems to be more suited for parameter sweeps.
    # I think the current approach is the most useful for our use case. We can easily add a loop over
    # the configs and submit an array of jobs.

    # NOTE: The AutoExecutor detects whether slurm is available. If not, it runs locally.
    # I want to test this a bit more to make sure it works properly with the paths and such.
    if config.slurm:
        slurm_executor = submitit.AutoExecutor(folder=f'{config.experiment.experiment_folder}/slurm')
        slurm_executor.update_parameters(
            **{**dataclasses.asdict(config.slurm), "slurm_job_name": config.experiment.experiment_id}
        )
        job = slurm_executor.submit(runner.run_experiment)
        logger.info(f'Submitted job `{job.job_id}` to Slurm with config {config.slurm}.')
    else:
        logger.info('Starting experiment locally.')
        runner.run_experiment()


if __name__ == "__main__":
    run_experiment()
