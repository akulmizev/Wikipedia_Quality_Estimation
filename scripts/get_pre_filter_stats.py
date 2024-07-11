import sys

from wqe import WikiLoader

URLS_TO_REMOVE = ["https://xh.wikipedia.org/wiki/Phi"]


def get_stats(language, stats_dir):

    stats_out = open(f"{stats_dir}/pre_filter_stats.{language}.txt", "w")
    stats_out.write("lang_id\tprocess\tn_chars\tn_docs\tfrac_chars\tfrac_docs\n")

    dataset = WikiLoader(language)
    dataset.load_dataset()
    dataset.pre_filter(num_proc=32, urls_to_remove=URLS_TO_REMOVE)

    stats_dict = {
        "current_chars": dataset.n_chars,
        "current_docs": dataset.n_docs,
        "full_chars": dataset.n_chars,
        "full_docs": dataset.n_docs,
        "total_chars": dataset.n_chars,
        "total_docs": dataset.n_docs
    }

    stats_out.write(f"{language}\traw\t{stats_dict['current_chars']}\t{stats_dict['current_docs']}\t{1.0}\t{1.0}\n")

    dataset.pre_filter(deduplicate_exact_match=True, num_proc=32)
    stats_dict = update_stats_dict(stats_dict, dataset.n_chars, dataset.n_docs)
    write_line(stats_out, language, "exact_match", stats_dict)

    dataset.pre_filter(char_cutoff=15, num_proc=32)
    stats_dict = update_stats_dict(stats_dict, dataset.n_chars, dataset.n_docs)
    write_line(stats_out, language, "char_cutoff_15", stats_dict)

    dataset.pre_filter(script_regex=True, char_cutoff=15, num_proc=32)
    stats_dict = update_stats_dict(stats_dict, dataset.n_chars, dataset.n_docs)
    write_line(stats_out, language, "script_regex", stats_dict)

    dataset.pre_filter(lang_id=False, deduplicate_min_hash=True, jaccard_threshold=0.85, num_proc=32)
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
    get_stats(lang, stats_dir)
    