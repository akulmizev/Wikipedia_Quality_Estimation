import datasets
from transformers import AutoModel
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi
from glob import glob
api = HfApi()

# files = glob("/lustre1/project/stg_00120/kushal/Wikipedia/Wikipedia_Quality_Estimation/WikiQuality/raw_wiki_lr1e03_warmup_grad_8/model/as/*")
# files = files.append(glob("/lustre1/project/stg_00120/kushal/Wikipedia/Wikipedia_Quality_Estimation/WikiQuality/raw_wiki_lr1e03_warmup_grad_8/model/as/*.safetensors"))
# print(files)

# api.create_repo(repo_id="WikiQuality/raw_wiki.as", private=True, exist_ok=True)

# for file in files:
#     api.upload_file(
#         path_or_fileobj=file,
#         repo_id="WikiQuality/raw_wiki.as",
#         path_in_repo="/"
        
#     )

file = "/lustre1/project/stg_00120/kushal/Wikipedia/Wikipedia_Quality_Estimation/WikiQuality/raw_wiki_lr1e03_warmup_grad_8/model/ur"
tokenizer = PreTrainedTokenizerFast.from_pretrained(file)
print(tokenizer)
model = AutoModel.from_pretrained(file)
tokenizer.push_to_hub("WikiQuality/raw_wiki.ur", private=True)

model.push_to_hub("WikiQuality/raw_wiki.ur", private=True)
# 
# api.upload_folder(
#     folder_path="/lustre1/project/stg_00120/kushal/Wikipedia/Wikipedia_Quality_Estimation/WikiQuality/raw_wiki_lr1e03_warmup_grad_8/model/as",
#     repo_id="WikiQuality/raw_wiki.as"
# )

# dataset_cfg = datasets.load_dataset("Wikipedia/Wikipedia_Quality_Estimation/WikiQuality/pre_filtered/ur")
# print(dataset_cfg)

# dataset_cfg.push_to_hub("WikiQuality/pre_filtered", config_name="ur", private=True)

# config = datasets.get_dataset_config_names("WikiQuality/pre_filtered")
# print(config)