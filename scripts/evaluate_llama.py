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


@hydra.main(version_base=None, config_path=None, config_name="config")
def run_experiment(config_dict: DictConfig) -> None:

    try:
        config_dict = OmegaConf.to_container(config_dict, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue:
        logger.error("Experiment config must be provided, e.g. `+experiment=basic`.")
        exit(1)


    partitions = ["pre_filtered", "raw_wiki"]
    languages = [("ha", "hau"), ("sw", "swa"), ("yo", "yor"), ("ig", "ibo")]
    shots = list(range(1,11))

    params = list(itertools.product(*[partitions, languages, shots]))

    for partition, (wiki_id, lang), n_shots in params:
        cfg = copy.deepcopy(config_dict)

        model_id = f"WikiQuality/continued_pretraining_{partition}.{wiki_id}"

        cfg['experiment']['wiki_id'] = wiki_id
        cfg['experiment']['experiment_id'] = f"{cfg['experiment']['experiment_id']}_{partition}"

        # Make sure to load the tokenizers from our peft models, not the base model. 
        cfg['tokenizer']['load_path'] = model_id

        # NOTE: lm_eval/load_path refers to the base model if you're using peft.
        # Peft weights come from somewhere else (model_id in this case)!
        cfg['lm_eval']['model_inference_config']['peft'] = model_id
        cfg['lm_eval']['num_fewshot'] = n_shots
        cfg['lm_eval']['tasks'] = [
            f"afrixnli_native_direct_{lang}"
            # f"afrimmlu_direct_{lang}",
        ]

        config  = MainConfig(**cfg)
        config.experiment.experiment_folder = f"{config.experiment.experiment_folder}/{partition}"

        runner = ExperimentRunner(config)
        slurm_executor = submitit.AutoExecutor(folder=f'{config.experiment.experiment_folder}/slurm')
        slurm_executor.update_parameters(
            **{**dataclasses.asdict(config.slurm), "slurm_job_name": config.experiment.experiment_id}
        )
        job = slurm_executor.submit(runner.run_experiment)
        logger.info(f'Submitted job `{job.job_id}` to Slurm with config {config.slurm}.')


if __name__ == "__main__":
    run_experiment()
