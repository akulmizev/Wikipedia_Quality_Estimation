from transformers import PreTrainedTokenizerFast

wikis = ['ig', 'ha', 'sw', 'pcm', 'yo']
partitions = [
    'all_methods_hi',
    'all_methods_lo',
    'length_hi',
    'length_lo',
    'unique_character_trigrams_hi',
    'unique_character_trigrams_lo',
    'unique_trigrams_hi',
    'unique_trigrams_lo',
    'unique_words_hi',
    'unique_words_lo'
]

for wiki in wikis:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(f'WikiQUality/pre_filtered.{wiki}')

    for partition in partitions:
        tokenizer.push_to_hub(f'WikiQuality/{partition}.{wiki}')