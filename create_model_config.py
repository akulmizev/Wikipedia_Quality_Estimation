from transformers import RobertaConfig, RobertaModel, AutoConfig
import json
import os


lang = "as"
vocab_mapper = json.load(open(f"./data/predicted_vocab.json", "r"))
vocab_size = vocab_mapper[lang]

config = RobertaConfig()
config.vocab_size = vocab_size
config.num_hidden_layers = 3
config.num_attention_heads = 5
config.hidden_size = 100
config.intermediate_size = 400
config.eos_token_id = 2
config.pad_token_id = 0
config.bos_token_id = 1

print(config)

if not os.path.exists(f"./config"):
    os.makedirs(f"./config")
config.save_pretrained(f"./config/{lang}")

# config = AutoConfig.from_pretrained(f"./config/{lang}")
# print(config)

