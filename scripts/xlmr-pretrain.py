import dataclasses
import logging
import itertools
import copy

import submitit
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from wqe.experiment.experiment import ExperimentRunner
from wqe.utils.config import MainConfig

cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These are all languages that meet the following requirements:
#   1. Not in XLM-R pretraining data
#   2. Has a pre_filtered section on the hub
#   3. Has a raw_wiki section on the hub
SAMPLE = [
    "ace",
    "awa",
    "ban",
    "bm",
    "ee",
    "fon",
    "ht",
    "ig",
    "ks",
    "lg",
    "lij",
    "ln",
    "mai",
    "mi",
    "mni",
    "nso",
    "ny",
    "pag",
    "pcm",
    "rn",
    "rw",
    "shn",
    "sn",
    "ti",
    "tn",
    "ts",
    "tw",
    "wo",
    "yo",
    "zu",
]


@hydra.main(version_base=None, config_path=None, config_name="config")
def run_experiment(config_dict: DictConfig) -> None:
    try:
        config_dict = OmegaConf.to_container(
            config_dict, resolve=True, throw_on_missing=True
        )
    except MissingMandatoryValue:
        logger.error("Experiment config must be provided, e.g. `+experiment=basic`.")
        exit(1)

    partitions = ["raw_wiki", "pre_filtered", "thresholded_wiki"]

    params = list(itertools.product(*[SAMPLE, partitions]))

    for wiki_id, partition in params:
        cfg = copy.deepcopy(config_dict)

        cfg["dataset"]["load_path"] = f"WikiQuality/{partition}"

        cfg["experiment"]["wiki_id"] = wiki_id
        cfg["experiment"]["experiment_id"] = f"xlmr_{partition}"
        config = MainConfig(**cfg)

        runner = ExperimentRunner(config)
        slurm_executor = submitit.AutoExecutor(
            folder=f"{config.experiment.experiment_folder}/slurm"
        )
        slurm_executor.update_parameters(
            **{
                **dataclasses.asdict(config.slurm),
                "slurm_job_name": config.experiment.experiment_id,
            }
        )
        job = slurm_executor.submit(runner.run_experiment)
        logger.info(
            f"Submitted job `{job.job_id}` to Slurm with config {config.slurm}."
        )
        if "debug" in config.slurm.slurm_partition:
            exit()


if __name__ == "__main__":
    run_experiment()
