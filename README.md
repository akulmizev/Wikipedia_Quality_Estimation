# Wikipedia_Quality_Estimation
This is the GitHub for the Wikipedia Quality Estimation Project. All code can be uploaded here. Feel free to create your own branches if necessary!

The idea here is to have a stream-lined easy-to-run repo which would conceptually follow this framework:
- partition.py (containing all partition functions)
- train.py (containing code to train tiny BERT/GPT per language per partition function)
- finetune.py (containing code to finetune each model on downstream tasks)
- evaluate.py
- run.sh (containing one slurm file that runs everything)

## To generate splits on partition functions:
`python partition.py --language [-l] as --partition [-p] stupid_filters` 
- Specify language in the ISO-2 format.
- Specify the partition function. Options:
  -   length
  -   unique_words
  -   unique_subwords
  -   unique_trigrams
  -   word_length
  -   english_chars
  -   stupid_filters (runs all of the above in one function)
- You can also specify whether you want to use the raw version of the wikiepdia, or the filtered one (where text has been filtered on a regex that remove latin chars in non-latin languages, and then further filtered using a language ID model) by:
 
`python partition.py -l <language> -p <partition> --filtered True/False`

It is set to `True` by default.


Running this script will create a `wikis/<language>/` directory where both your high_quality and low_quality partitions will be saved.

## To see stats per partition function, run:
`python partition.py -l <language> -p stats`
This will create a directory called 'stats' and save csv files for every partition function, which you can then explore in the jupyter notebook `stats.ipynb`.

## To add new partition functions:
The class `Partition()` loads the specified wikipedia upon calling, and creates two variables - `self.language` (containing the language code) and `self.dataset` (containing the huggingface dataset). These are accessible to all the functions contained in `class Partition()`. To add your own partition, define a function in the class partition that returns a string of high_quality and low_quality examples separated by a newline. That function can then be called under main(). 
