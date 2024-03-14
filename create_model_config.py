from transformers import RobertaConfig, RobertaModel, AutoConfig, AutoModelForMaskedLM, BertConfig, BertModel
import json
import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--lang", default=None, type=str, required=True,
#                     help="specify the language for which the model configuration is to be created. Needs to be in ISO-2 format")
# parser.add_argument("--output_dir", default="./config", type=str, required=False)
# parser.add_argument('--base_model', default="roberta", type=str, required=False)



lang = "as"
vocab_mapper = json.load(open(f"./data/predicted_vocab.json", "r"))
vocab_size = vocab_mapper[lang]

# config = RobertaConfig()
# config.base_model = "roberta"
# config.vocab_size = vocab_size
# config.num_hidden_layers = 3
# config.num_attention_heads = 5
# config.hidden_size = 100
# config.intermediate_size = 400
# config.pad_token_id = -1
# config.tokenizer_class = "PreTrainedTokenizerFast"
# config.max_position_embeddings = 512

config = BertConfig()
config.base_model = "bert"
config.vocab_size = vocab_size
config.num_hidden_layers = 2
config.hidden_size = 128
config.intermediate_size = 512
config.num_attention_heads = 4
config.max_position_embeddings = 512



#
if not os.path.exists(f"./config"):
    os.makedirs(f"./config")
config.save_pretrained(f"./config/{lang}/config_bert")

# config = AutoConfig.from_pretrained(f"./config/{lang}")
# print(config)

# model = AutoModelForMaskedLM.from_config(config=config)
# print(model.get_input_embeddings().weight.shape[0])
