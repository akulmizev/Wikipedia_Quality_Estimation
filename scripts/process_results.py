import os
import sys

SCORES_DIR = sys.argv[1]

def main():

    with open("all_scores.csv", "w") as outfile:
        outfile.write("metric,score,lang,partition,task\n")
        for partition in os.listdir(SCORES_DIR):
            for lang in ["pcm", "ha", "ig", "sw", "yo"]:
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
                                outfile.write(f"{metric},{score},{lang},{partition},{task}\n")
                except FileNotFoundError:
                    continue

if __name__ == "__main__":
    main()
