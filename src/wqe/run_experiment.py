import argparse
import os
import yaml

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, required=True,
                        help="Specify path to configuration file.")
    parser.add_argument("--wiki_id", default=None, type=str, required=True,
                        help="Specify Wiki ID for extracting Wiki dump.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        config["experiment"].update({"wiki_id": args.wiki_id})

    experiment_cfg, data_cfg, tokenizer_cfg, pretrain_cfg, finetune_cfg = parse_config(config)

    if experiment_cfg.local_path:
        experiment_path = f"{experiment_cfg.local_path}/{experiment_cfg.experiment_id}"
        os.makedirs(experiment_path, exist_ok=True)

    if experiment_cfg.hub_path:
        api = HfApi()

    if data_cfg:
        dataset = WikiLoader(data_cfg, wiki_id=experiment_cfg.wiki_id)
        if data_cfg.pre_filter:
            dataset.do_pre_filter()
        if data_cfg.partition:
            dataset.apply_partition()
        if data_cfg.split:
            dataset.generate_splits()
        if data_cfg.export:
            dataset.save(f"{experiment_path}/data/{experiment_cfg.wiki_id}")
            if data_cfg.push_to_hub:
                data = load_dataset(f"{experiment_path}/data/{experiment_cfg.wiki_id}")
                data.push_to_hub(f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}", 
                                 config_name=f"{experiment_cfg.wiki_id}", 
                                 private=True)

    if tokenizer_cfg:
        if tokenizer_cfg.load.method == "config":
            tokenizer = PreTrainedTokenizerFast.train_from_config(
                dataset["train"],
                config=tokenizer_cfg.parameters,
                batch_size=1000
            )
            # tokenizer = SentencePieceTokenizer.train_from_iterator(
            #     self=SentencePieceTokenizer(config=tokenizer_cfg.parameters),
            #     config=tokenizer_cfg.parameters,
            #     dataset=dataset['train']
            # )
            if tokenizer_cfg.export:
                tokenizer.save_pretrained(f"{experiment_path}/model/{experiment_cfg.wiki_id}")
        elif tokenizer_cfg.load.method == "hub":
            tokenizer = AutoTokenizer.from_pretrained(f"{tokenizer_cfg.load.path}.{experiment_cfg.wiki_id}")
        if tokenizer_cfg.push_to_hub:
            tokenizer.push_to_hub(f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}.{experiment_cfg.wiki_id}", private=True)

    if pretrain_cfg:
        export_path = f"{experiment_path}/model/{experiment_cfg.wiki_id}" if pretrain_cfg.export else None
        model = MLM(
            pretrain_cfg.training_parameters,
            tokenizer,
            load_method=pretrain_cfg.load.method,
            load_path=pretrain_cfg.load.path,
            export_path=export_path
            )
        if experiment_cfg.wandb_entity:
            model.init_wandb(
                project=f"{experiment_cfg.experiment_id}.{experiment_cfg.wiki_id}",
                entity=experiment_cfg.wandb_entity,
                parameters=pretrain_cfg.training_parameters
                )
        if pretrain_cfg.do_train:
            model.train(dataset)
        
        if pretrain_cfg.test_data:
            test_dataset = WikiLoader.load_dataset_directly(import_config=pretrain_cfg.test_data,
                                                            wiki_id=experiment_cfg.wiki_id)
            model.test(dataset=test_dataset)

        if pretrain_cfg.push_to_hub:
            model = AutoModel.from_pretrained(f"{export_path}")
            model.push_to_hub(f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}.{experiment_cfg.wiki_id}", private=True)

    if finetune_cfg:
        try:
            finetune_dataset = load_dataset(finetune_cfg.dataset_path, experiment_cfg.wiki_id)
        except:
            raise ValueError("Dataset not found. Please specify a valid dataset path.")

        if finetune_cfg.task == "ner":
            label_set = finetune_dataset["train"].features["tags"].feature.names
            model = Tagger(
                finetune_cfg.training_parameters,
                label_set=label_set,
                load_path=f"{finetune_cfg.load.path}.{experiment_cfg.wiki_id}"
            )
        elif finetune_cfg.task == "pos":
            label_set = finetune_dataset["train"].features["tags"].feature.names
            model = Tagger(
                finetune_cfg.training_parameters,
                label_set=label_set,
                load_path=f"{finetune_cfg.load.path}.{experiment_cfg.wiki_id}"
            )
        elif finetune_cfg.task == "sentiment_analysis":
            label_set = finetune_dataset["train"].features["label"].names
            model = Classifier(
                finetune_cfg.training_parameters,
                label_set=label_set,
                load_path=f"{finetune_cfg.load.path}.{experiment_cfg.wiki_id}"
            )
        if experiment_cfg.wandb_entity:
            model.init_wandb(
                project=f"{experiment_cfg.experiment_id}.{experiment_cfg.wiki_id}",
                entity=experiment_cfg.wandb_entity,
                parameters=finetune_cfg.training_parameters
            )
        if finetune_cfg.do_train:
            model.train(finetune_dataset)

        # model.train(dataset)
        # model.save(f"{experiment_path}/model/{experiment_cfg.wiki_id}")
        # if api.repo_exists(f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}"):
        #     api.create_repo(
        #         repo_id=f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}",
        #         repo_type="model",
        #         private=False
        #     )
        # api.upload_folder(
        #     folder_path=f"{experiment_path}/model/",
        #     repo_id=f"{experiment_cfg.hub_path}/{experiment_cfg.experiment_id}",
        #     repo_type="model"
        # )
        #     if config["pretrain"]["train"]:
        #         model.train(dataset)
        #     if "test_data" in config["pretrain"]:
        #         test_dataset = WikiDatasetFromConfig.load_dataset_directly(
        #             config["pretrain"]["test_data"],
        #             wiki_id=args.wiki_id, split="test"
        #         )
        #         model.test(test_dataset)


        # elif "from_pretrained" in tokenizer_cfg:
        #     tokenizer = FastTokenizerFromConfig.from_pretrained(tokenizer_cfg["from_pretrained"])
    # if "tokenizer" in config:
    #     if "from_config" in config["tokenizer"]:
    #         tokenizer = FastTokenizerFromConfig.train_from_config(
    #             dataset["train"],
    #             config_path=config["tokenizer"]["from_config"],
    #             batch_size=1000)
    #         if "export" in config["tokenizer"]:
    #             tokenizer.save()
    #     elif "from_pretrained" in config["tokenizer"]:
    #         tokenizer = FastTokenizerFromConfig.from_pretrained(config["tokenizer"]["from_pretrained"])
    #
    # if "pretrain" in config:
    #     model = WikiMLM(config, tokenizer, dataset)
    #     if config["pretrain"]["train"]:
    #         model.train(dataset)
    #     if "test_data" in config["pretrain"]:
    #         test_dataset = WikiDatasetFromConfig.load_dataset_directly(
    #             config["pretrain"]["test_data"],
    #             wiki_id=args.wiki_id, split="test"
    #         )
    #         model.test(test_dataset)

    # if "finetune" in config:
    #     ner_dataset = load_dataset("indic_glue", f"wiki-ner.{config['wiki_id']}")
    #     model = WikiNER(config["finetune"])
    #     model.prepare_model(ner_dataset, fast_tokenizer)
    #     model.train()

    pass


if __name__ == "__main__":
    main()
