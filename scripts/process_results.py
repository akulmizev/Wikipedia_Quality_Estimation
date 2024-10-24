import os
import sys
import itertools
from glob import glob

# SCORES_DIR = sys.argv[1]
SCORES_DIR = "./slurm/experiments"

SAMPLE = [
     "pcm",
    "ha",
    "ig",
    "sn",
    "so",
    "rw",
    "tw",
    "ln",
    "wo",
    "om",
    "ny",
    "ee",
    "bm",
    "ts",
    "fon",
    "ti",
    "sw",
    "am", 
    "zu", #zu left
    "ary",
    "lg",
    "tn",
    "rn",
    # "hi",
    "ckb",
    "ks",
    "awa",
    "yo",
    "kk",
    "mr",
    "te",
    "ht",
    "nso",
    "lij",
    "ace",
    "shn",
    "mai", #mai left
    "ban",
    "mni",
    "mi",
    "pag",
    "vi",
    "war",
    "tt",
    "azb",
    "ro",
    "id", #id left
    "hu",
    "nl",
    "uk",
    "ceb",
    "min",
    # "sv",
]
partitions = ["raw_wiki", "pre_filtered", "thresholded_wiki",
                  "absolute_lo", "absolute_hi",
                  "fraction_lo", "fraction_hi",
                  "entropy_lo", "entropy_hi", "baseline"]
params = list(itertools.product(*[SAMPLE, partitions]))
def main():

    with open("all_scores.csv", "w") as outfile:
        outfile.write("score,lang,partition,task\n")
        for partition in os.listdir(SCORES_DIR):
            for lang in SAMPLE:
                try:
                    for scores in os.listdir(f"{SCORES_DIR}/{partition}/{lang}"):
                        try:
                            task = scores.split(".")[1]
                        except:
                            continue
                        with open(os.path.join(f"{SCORES_DIR}/{partition}/{lang}", scores), "r") as infile:
                            for line in infile.readlines():
                                line = line.strip()
                                metric = line.split("\t")[0]
                                score = line.split("\t")[1]
                                if metric == 'f1':
                                    outfile.write(f"{score},{lang},{partition},{task}\n")
                except FileNotFoundError:
                    continue

def main_2():
    
    with open("ner-4.csv", "w") as outfile:
        outfile.write("score,lang,partition,task\n")
        for wiki_id, partition in params:
            folder = f"{SCORES_DIR}/{partition}.ner.4/{wiki_id}"
            file = glob(f"{folder}/*.txt")
            if not file:
                continue
            task = "ner"
            with open (file[0], "r") as infile:
                for line in infile.readlines():
                    line = line.strip()
                    metric = line.split("\t")[0]
                    score = line.split("\t")[1]
                    if metric == "f1":
                        outfile.write(f"{score},{wiki_id},{partition},{task}\n")

def main_3():

    with open("roberta_tokmerge.csv", "w") as outfile:
        outfile.write("score,lang,partition,task\n")
        for wiki_id, partition in params:
            folder = f"{SCORES_DIR}/roberta_eval_{partition}_tokmerge/{wiki_id}"
            file = glob(f"{folder}/*.txt")
            if not file:
                continue
            task = "sib200"
            with open (file[0], "r") as infile:
                for line in infile.readlines():
                    line = line.strip()
                    metric = line.split("\t")[0]
                    score = line.split("\t")[1]
                    if metric == "f1":
                        outfile.write(f"{score},{wiki_id},{partition},{task}\n")


if __name__ == "__main__":
    main_2()