from wqe import WikiLoader

# LANGS = [
#     'sw', 'ha', 'yo', 'ig', 'am',
#     'sn', 'zu', 'ary', 'so', 'rw',
#     'tw', 'ln', 'lg', 'xh', 'wo',
#     'om', 'tn', 'ny', 'pcm', 'ee',
#     'bm', 'ts', 'rn', 'fon', 'ti'
# ]

# LANGS = [
#     'sw', 'ha', 'yo', 'ig', 'am',
#     'sn', 'zu', 'ary', 'so', 'rw',
#     'tw', 'ln', 'lg', 'wo',
#     'om', 'tn', 'ny', 'pcm', 'ee',
#     'bm', 'ts', 'rn', 'fon', 'ti'
# ]

LANGS = ["xh"]
URLS_TO_REMOVE = ["https://xh.wikipedia.org/wiki/Phi"]


def get_stats(language):
    stats_out = open(f"/home/akulmizev/Repos/Wikipedia_Quality_Estimation/stats/pre_filter_stats.{language}.txt", "w")
    stats_out.write("lang_id\tprocess\tn_chars\tn_docs\tfrac_chars\tfrac_docs\n")

    dataset = WikiLoader(language)
    dataset.load_dataset()
    dataset.pre_filter(num_proc=32, urls_to_remove=URLS_TO_REMOVE)

    current_chars = dataset.n_chars
    current_docs = dataset.n_docs
    full_chars = current_chars
    full_docs = current_docs
    total_chars = full_chars
    total_docs = full_docs
    stats_out.write(f"{language}\traw\t{current_chars}\t{current_docs}\t{1.0}\t{1.0}\n")

    dataset.pre_filter(deduplicate_exact_match=True, num_proc=32)
    deleted_chars = total_chars - dataset.n_chars
    deleted_docs = total_docs - dataset.n_docs
    frac_chars_deleted = deleted_chars / full_chars
    frac_docs_deleted = deleted_docs / full_docs
    total_chars -= deleted_chars
    total_docs -= deleted_docs
    stats_out.write(
        f"{language}\texact_match\t{deleted_chars}\t{deleted_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")

    dataset.pre_filter(char_cutoff=15, num_proc=32)
    deleted_chars = total_chars - dataset.n_chars
    deleted_docs = total_docs - dataset.n_docs
    frac_chars_deleted = deleted_chars / full_chars
    frac_docs_deleted = deleted_docs / full_docs
    total_chars -= deleted_chars
    total_docs -= deleted_docs
    stats_out.write(
        f"{language}\tchar_cutoff_15\t{deleted_chars}\t{deleted_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")

    dataset.pre_filter(script_regex=True, char_cutoff=15, num_proc=32)
    deleted_chars = total_chars - dataset.n_chars
    deleted_docs = total_docs - dataset.n_docs
    frac_chars_deleted = deleted_chars / full_chars
    frac_docs_deleted = deleted_docs / full_docs
    total_chars -= deleted_chars
    total_docs -= deleted_docs
    stats_out.write(
        f"{language}\tscript_regex\t{deleted_chars}\t{deleted_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")

    # dataset.pre_filter(lang_id=True, char_cutoff=15)
    # current_chars = total_chars - dataset.n_chars
    # current_docs = total_docs - dataset.n_docs
    # frac_chars_deleted = current_chars / full_chars
    # frac_docs_deleted = current_docs / full_docs
    # total_chars = dataset.n_chars
    # total_docs = dataset.n_docs
    # stats_out.write(f"{lang}\tscript_regex\t{current_chars}\t{current_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")

    dataset.pre_filter(
        lang_id=False,
        deduplicate_min_hash=True,
        jaccard_threshold=0.85,
        num_proc=32
    )
    deleted_chars = total_chars - dataset.n_chars
    deleted_docs = total_docs - dataset.n_docs
    frac_chars_deleted = deleted_chars / full_chars
    frac_docs_deleted = deleted_docs / full_docs
    total_chars -= deleted_chars
    total_docs -= deleted_docs
    stats_out.write(f"{language}\tjaccard_085\t{deleted_chars}\t{deleted_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")

    # dataset.pre_filter(
    #     deduplicate_min_hash=True,
    #     jaccard_threshold=0.7,
    #     num_proc=32,
    # )
    # deleted_chars = total_chars - dataset.n_chars
    # deleted_docs = total_docs - dataset.n_docs
    # frac_chars_deleted = deleted_chars / full_chars
    # frac_docs_deleted = deleted_docs / full_docs
    # total_chars -= deleted_chars
    # total_docs -= deleted_docs
    # stats_out.write(f"{language}\tjaccard_07\t{deleted_chars}\t{deleted_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")
    #
    # dataset.pre_filter(
    #     deduplicate_min_hash=True,
    #     jaccard_threshold=0.5,
    #     num_proc=32
    # )
    # deleted_chars = total_chars - dataset.n_chars
    # deleted_docs = total_docs - dataset.n_docs
    # frac_chars_deleted = deleted_chars / full_chars
    # frac_docs_deleted = deleted_docs / full_docs
    # total_chars -= deleted_chars
    # total_docs -= deleted_docs
    # stats_out.write(f"{language}\tjaccard_05\t{deleted_chars}\t{deleted_docs}\t{frac_chars_deleted}\t{frac_docs_deleted}\n")

    stats_out.write(
        f"{language}\tremaining\t{total_chars}\t{total_docs}\t{total_chars / full_chars}\t{total_docs / full_docs}\n")
    stats_out.close()


if __name__ == "__main__":
    for lang in LANGS:
        get_stats(lang)
    