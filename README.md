# Wikipedia_Quality_Estimation
This is the GitHub for the Wikipedia Quality Estimation Project. All code can be uploaded here. Feel free to create your own branches if necessary!

The idea here is to have a stream-lined easy-to-run repo which would conceptually follow this framework:
- partition.py (containing all partition functions)
- train_adapter.py (containing code to train adapters per language per partition function)
- train_tiny_bert.py (containing code to train tiny BERT per language per partition function)
- finetune.py (containing code to finetune each model on downstream tasks)
- run.sh (containing one slurm file that runs everything)

