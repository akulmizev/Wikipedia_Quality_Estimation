from transformers import RobertaConfig, RobertaModel, AutoConfig, AutoModelForMaskedLM, BertConfig, BertModel, DebertaConfig
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default=None, type=str, required=False,
                    help="specify the language for which the model configuration is to be created. Needs to be in ISO-2 format")


args = parser.parse_args()

# lang = "ur"
# vocab_mapper = json.load(open(f"../wqe/data/resources/wiki_mappings.json", "r"))
# vocab_size = vocab_mapper[lang]

config = DebertaConfig()
config.base_model = "deberta"
# config.vocab_size = vocab_size
config.num_hidden_layers = 4
config.hidden_size = 312
config.intermediate_size = 1200
config.num_attention_heads = 12
config.max_position_embeddings = 512


if not os.path.exists(f"../config"):
    os.makedirs(f"../config")
config.save_pretrained(f"../config/model/tiny_deberta")
