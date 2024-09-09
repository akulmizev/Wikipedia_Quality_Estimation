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


    models = ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-2-7b-hf"]
    languages = [("ha", "hau"), ("sw", "swa"), ("yo", "yor"), ("ig", "ibo")]
    shots = list(range(1,11))

    params = list(itertools.product(*[languages, shots, models]))

    for (wiki_id, lang), n_shots, model_id in params:
        cfg = copy.deepcopy(config_dict)

        model_name = model_id.split('/')[-1].split('-')[0].lower()

        cfg['experiment']['wiki_id'] = wiki_id
        cfg['experiment']['experiment_id'] = f"baseline-{model_name}"

        cfg['tokenizer']['load_path'] = model_id

        cfg['lm_eval']['load_path'] = model_id
        cfg['lm_eval']['num_fewshot'] = n_shots
        cfg['lm_eval']['log_samples'] = True
        cfg['lm_eval']['model_inference_config']['peft'] = None
        cfg['lm_eval']['tasks'] = [
            f"afrixnli_native_direct_{lang}",
            f"afrimmlu_direct_{lang}",
        ]

        config  = MainConfig(**cfg)

        runner = ExperimentRunner(config)
        slurm_executor = submitit.AutoExecutor(folder=f'{config.experiment.experiment_folder}/slurm')
        slurm_executor.update_parameters(
            **{**dataclasses.asdict(config.slurm), "slurm_job_name": config.experiment.experiment_id}
        )
        job = slurm_executor.submit(runner.run_experiment)
        logger.info(f'Submitted job `{job.job_id}` to Slurm with config {config.slurm}.')

if __name__ == "__main__":
    run_experiment()
