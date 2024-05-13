import datasets

#UD
# naija_ud = datasets.load_dataset("universal_dependencies", "pcm_nsc")
# print(naija_ud)
# urdu_ud = datasets.load_dataset("universal_dependencies", "ur_udtb")
# print(urdu_ud)
#
# naija_ud.push_to_hub("WikiQuality/universal_dependencies", config_name="pcm", private=True)
# urdu_ud.push_to_hub("WikiQuality/universal_dependencies", config_name="ur", private=True)

#NER
# naija_ner = datasets.load_dataset("masakhane/masakhaner2", "pcm")
# print(naija_ner)
# swahili_ner = datasets.load_dataset("masakhane/masakhaner2", "swa")
# print(swahili_ner)
# assamese_ner = datasets.load_dataset("ai4bharat/naamapadam", "as")
# print(assamese_ner)
# urdu_ner = datasets.load_dataset("wikiann", "ur")
# print(urdu_ner)
#
# naija_ner.push_to_hub("WikiQuality/named_entity", config_name="pcm", private=True)
# swahili_ner.push_to_hub("WikiQuality/named_entity", config_name="sw", private=True)
# assamese_ner.push_to_hub("WikiQuality/named_entity", config_name="as", private=True)
# urdu_ner.push_to_hub("WikiQuality/named_entity", config_name="ur", private=True)

#Sentiment
naija_senti = datasets.load_dataset("HausaNLP/AfriSenti-Twitter", "pcm")
naija_senti = naija_senti.rename_column("tweet", "text")
print(naija_senti)

swahili_senti = datasets.load_dataset("HausaNLP/AfriSenti-Twitter", "swa")
swahili_senti = swahili_senti.rename_column("tweet", "text")
print(swahili_senti)

# urdu_senti_train = datasets.load_dataset("Juanid14317/UrduSentimentAnalysis")["train"]
# print(urdu_senti_train)
# urdu_senti = urdu_senti_train.train_test_split(test_size=0.1, shuffle=True, seed=42)
# urdu_senti = datasets.DatasetDict({
#     "train": urdu_senti["train"],
#     "validation": urdu_senti["test"],
#     "test": datasets.load_dataset("Juanid14317/UrduSentimentAnalysis")["test"]
# })
# print(urdu_senti)
#
# print(len(urdu_senti["train"]), len(urdu_senti["validation"]), len(urdu_senti["test"]))

naija_senti.push_to_hub("WikiQuality/sentiment_analysis", config_name="pcm", private=True)
swahili_senti.push_to_hub("WikiQuality/sentiment_analysis", config_name="sw", private=True)
# urdu_senti.push_to_hub("WikiQuality/sentiment_analysis", config_name="ur", private=True)


