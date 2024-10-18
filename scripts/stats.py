import sys
import importlib.resources as pkg_resources
from wqe import WikiLoader
from wqe.data import resources
import os
import json
import re

URLS_TO_REMOVE = ["https://xh.wikipedia.org/wiki/Phi"]


def get_stats(language, stats_dir):

    stats_out = open(f"{stats_dir}/pre_filter_stats.{language}.txt", "w")
    stats_out.write("lang_id\tprocess\tn_chars\tn_docs\tfrac_chars\tfrac_docs\n")

    dataset = WikiLoader(language)
    dataset.load_dataset()
    # dataset.pre_filter(num_proc=32, urls_to_remove=URLS_TO_REMOVE)

    stats_dict = {
        "current_chars": dataset.n_chars,
        "current_docs": dataset.n_docs,
        "full_chars": dataset.n_chars,
        "full_docs": dataset.n_docs,
        "total_chars": dataset.n_chars,
        "total_docs": dataset.n_docs
    }

    stats_out.write(f"{language}\traw\t{stats_dict['current_chars']}\t{stats_dict['current_docs']}\t{1.0}\t{1.0}\n")

    dataset.pre_filter(script_regex=True, urls_to_remove=URLS_TO_REMOVE)
    stats_dict = update_stats_dict(stats_dict, dataset.n_chars, dataset.n_docs)
    write_line(stats_out, language, "script_regex", stats_dict)

    dataset.deduplicate(exact_match=True, n_shingles=1)
    stats_dict = update_stats_dict(stats_dict, dataset.n_chars, dataset.n_docs)
    write_line(stats_out, language, "exact_match", stats_dict)

    dataset.deduplicate(min_hash=True, n_shingles=1)
    stats_dict = update_stats_dict(stats_dict, dataset.n_chars, dataset.n_docs)
    write_line(stats_out, language, "jaccard_085", stats_dict)

    out = [
        language,
        "remaining",
        stats_dict["total_chars"],
        stats_dict["total_docs"],
        stats_dict["total_chars"] / stats_dict["full_chars"],
        stats_dict["total_docs"] / stats_dict["full_docs"]
    ]

    out_line = "\t".join(map(str, out)) + "\n"
    stats_out.write(out_line)
    stats_out.close() 


def update_stats_dict(stats_dict, n_chars, n_docs):

    stats_dict["deleted_chars"] = stats_dict["total_chars"] - n_chars
    stats_dict["deleted_docs"] = stats_dict["total_docs"] - n_docs
    stats_dict["frac_chars_deleted"] = stats_dict["deleted_chars"] / stats_dict["full_chars"]
    stats_dict["frac_docs_deleted"] = stats_dict["deleted_docs"] / stats_dict["full_docs"]
    stats_dict["total_chars"] -= stats_dict["deleted_chars"]
    stats_dict["total_docs"] -= stats_dict["deleted_docs"]

    return stats_dict


def write_line(stats_out, lang, process, stats_dict):

    out = [
        lang,
        process,
        stats_dict["deleted_chars"],
        stats_dict["deleted_docs"],
        stats_dict["frac_chars_deleted"],
        stats_dict["frac_docs_deleted"]
    ]

    out_line = "\t".join(map(str, out)) + "\n"

    stats_out.write(out_line)


if __name__ == "__main__":
    lang = sys.argv[1]
    stats_dir = sys.argv[2]
    with pkg_resources.open_text(resources, 'wiki_mappings.json') as file:
        wiki_mappings = json.load(file)
    wikis = [wiki_id for wiki_id, _ in wiki_mappings.items()]

    if lang == 'all_wikis_1':
        
        # wikis_done = os.listdir("/lustre1/project/stg_00120/kushal/Wikipedia_Quality_Estimation/stats")
        # regex = re.compile(r'.*?\.(.*?)\.txt')
        # wikis_done = [regex.findall(wiki)[0] for wiki in wikis_done]

        # for wiki_id, item in wiki_mappings.items():
        #     if wiki_id not in wikis_done:
        #         get_stats(wiki_id, stats_dir)

        wikis = wikis[:10]

    elif lang == 'all_wikis_2':

        wikis = wikis[10:50]

    elif lang == 'all_wikis_3':

        wikis = wikis[50:100]

    elif lang == "all_wikis_4":

        wikis = wikis[100:150]

    elif lang == "all_wikis_5":

        wikis = wikis[150:200]

    elif lang == "all_wikis_6":

        wikis = wikis[200:]

    else:
        
        wikis = [lang]
    
    for wiki in wikis:
        get_stats(wiki, stats_dir)