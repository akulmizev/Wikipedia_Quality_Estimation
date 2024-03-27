from datasets import Dataset
from datasets import load_dataset
#load dataset from parquet file
ur = load_dataset('WikiQuality/pcm_filtered')
print(ur)
for example in ur:
    print(example)
    break
# print(eng)

# match = Dataset.from_parquet('ur_wiki_langid_match.parquet')
# # print(match)
#
# eng_urls = eng['url']
# match_urls = match['url']
#
# eng_text = eng['text']
# match_text = match['text']
#
# eng_id = eng['id']
# match_id = match['id']
#
# # eng_dict = {eng_urls[i]: eng_text[i] for i in range(len(eng_urls))}
# # match_dict = {match_urls[i]: match_text[i] for i in range(len(match_urls))}
#
# eng_dict = {eng_id[i]: eng_text[i] for i in range(len(eng_id))}
# match_dict = {match_id[i]: match_text[i] for i in range(len(match_id))}
# n = 0
# longer_match = {}
# for k, v in match_dict.items():
#     if len(v) < len(eng_dict[k]):
#         longer_match[k] = v
#
#
# # print(len(longer_match))
# # print(longer_match.keys())
# # for k, v in longer_match.items():
#     # print(k, v)
#     # print("+====================+")
#     # print(eng_dict[k])
#     # print("#####################")
#     # with open(f'match_english/{k}_match.txt', 'w') as f:
#     #     f.write(v)
#     # with open(f'match_english/{k}_eng.txt', 'w') as f:
#     #     f.write(eng_dict[k])
#
#
#
#
# match_text = " ".join(match_text)
# eng_text = " ".join(eng_text)
# print(len(match_text), len(eng_text))
# print(len(eng_text) - len(match_text))