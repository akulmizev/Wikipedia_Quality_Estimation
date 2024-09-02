# Quality Signal Metrics



| Annotation ID | Description                                    | Higher is Better |
|---------------|------------------------------------------------|-------------------|
| `length_chars` | Number of characters                           | Yes |
| `length_words` | Number of words                                | Yes |
| `unique_words` | Number of unique words                         | Yes |
| `unique_trigrams` | Number of unique word trigrams                 | Yes |
| `unique_chars` | Number of unique characters                    | Yes |
| `unique_char_trigrams` | Number of unique character trigrams            | Yes |
| `alpha_chars` | Number of alphabetic characters                | Yes |
| `frac_all_caps_words` | Fraction of words that are all uppercase       | Yes |
| `frac_symbol_to_words` | Fraction of words that are symbols (#, ..., …) | No |
| `frac_pipes_in_words` | Fraction of words containing pipe symbols (\|) | No |
| `frac_no_script_words` | Fraction of words without script characters    | Yes |
| `doc_mean_word_length` | Average word length                            | Yes |
| `frac_unique_words` | Fraction of unique words                       | Yes |
| `frac_unique_chars` | Fraction of unique characters                  | Yes |
| `frac_unique_trigrams` | Fraction of unique word trigrams               | Yes |
| `unigram_entropy` | Entropy of word unigrams                       | Yes |
| `trigram_entropy` | Entropy of word trigrams                       | Yes |
| `char_entropy` | Entropy of characters                          | Yes |
| `lines_end_with_punctuation` | Fraction of lines ending with punctuation      | Yes |
| `frac_lines_end_ellipsis` | Fraction of lines ending with ellipsis (...)   | No |
| `num_lines` | Number of lines                                | Yes |
| `num_words_per_line` | Average number of words per line               | Yes |
| `perplexity` | Perplexity using a language model              | No |