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
    # "pcm", IS NOT IN SIB-200
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

lang2id = {
    "ha": "hau_Latn",
    "ig": "ibo_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "rw": "kin_Latn",
    "tw": "twi_Latn",
    "ln": "lin_Latn",
    "wo": "wol_Latn",
    "ny": "nya_Latn",
    "ee": "ewe_Latn",
    "bm": "bam_Latn",
    "ts": "tso_Latn",
    "fon": "fon_Latn",
    "ti": "tir_Ethi",
    "om": "gaz_Latn",
    "sw": "swh_Latn",
    "zu": "zul_Latn",
    "ary": "ary_Arab",
    "lg": "lug_Latn",
    "tn": "tsn_Latn",
    "rn": "run_Latn",
    "am": "amh_Ethi",
    "hi": "hin_Deva",
    "ckb": "ckb_Arab",
    "ks": "kas_Arab",
    "awa": "awa_Deva",
    "yo": "yor_Latn",
    "kk": "kaz_Cyrl",
    "mr": "mar_Deva",
    "te": "tel_Telu",
    "ht": "hat_Latn",
    "nso": "nso_Latn",
    "lij": "lij_Latn",
    "ace": "ace_Arab",
    "shn": "shn_Mymr",
    "mai": "mai_Deva",
    "ban": "ban_Latn",
    "mni": "mni_Beng",
    "mi": "mri_Latn",
    "pag": "pag_Latn",
}


def run_job(config):
    runner = ExperimentRunner(config)
    slurm_executor = submitit.AutoExecutor(
        folder=f"{config.experiment.experiment_folder}/slurm"
    )
    slurm_executor.update_parameters(
        **{
            **dataclasses.asdict(config.slurm),
            "slurm_job_name": f"{config.experiment.experiment_id}.{config.experiment.wiki_id}",
        }
    )
    job = slurm_executor.submit(runner.run_experiment)
    logger.info(f"Submitted job `{job.job_id}` to Slurm with config {config.slurm}.")


@hydra.main(version_base=None, config_path=None, config_name="config")
def run_experiment(config_dict: DictConfig) -> None:
    try:
        config_dict = OmegaConf.to_container(
            config_dict, resolve=True, throw_on_missing=True
        )
    except MissingMandatoryValue:
        logger.error("Experiment config must be provided, e.g. `+experiment=basic`.")
        exit(1)

    partitions = ["raw_wiki", "pre_filtered"]
    params = list(itertools.product(*[SAMPLE, partitions]))

    for wiki_id, partition in params:
        cfg = copy.deepcopy(config_dict)

        cfg["finetune"]["load_path"] = f"experiments/xlmr_{partition}/{wiki_id}/model"
        cfg["finetune"]["dataset_config"] = lang2id[wiki_id]

        cfg["tokenizer"]["load_path"] = "FacebookAI/xlm-roberta-base"

        cfg["experiment"]["wiki_id"] = wiki_id
        cfg["experiment"]["experiment_id"] = f"xlmr_eval_{partition}"

        config = MainConfig(**cfg)
        run_job(config)
        if "debug" in config.slurm.slurm_partition:
            exit()

    for wiki_id in SAMPLE:
        cfg = copy.deepcopy(config_dict)

        cfg["finetune"]["load_path"] = "FacebookAI/xlm-roberta-base"
        cfg["finetune"]["dataset_config"] = lang2id[wiki_id]

        cfg["tokenizer"]["load_path"] = "FacebookAI/xlm-roberta-base"

        cfg["experiment"]["wiki_id"] = wiki_id
        cfg["experiment"]["experiment_id"] = "xlmr_eval_baseline"

        config = MainConfig(**cfg)
        run_job(config)


if __name__ == "__main__":
    run_experiment()
