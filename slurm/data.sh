#!/bin/bash

python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_low.yaml --wiki_id pcm
python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_high.yaml --wiki_id pcm

python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_low.yaml --wiki_id ur
python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_high.yaml --wiki_id ur

python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_low.yaml --wiki_id as
python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_high.yaml --wiki_id as

python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_low.yaml --wiki_id sw
python wqe/run_experiment.py --config config/experiment/make_partitioned_wiki_high.yaml --wiki_id sw
