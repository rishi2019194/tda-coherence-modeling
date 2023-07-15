'''
Read the .csv files of WSJ and GCDC. Extract balanced subsets for evaluating few-shot learning.
'''
import pandas as pd 
import json
from math import ceil

few_shot_percentages = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75]

## GCDC

clinton = pd.read_csv("GCDC_Dataset/clinton_train.csv", index_col=0)
yahoo   = pd.read_csv("GCDC_Dataset/yahoo_train.csv", index_col=0)
yelp    = pd.read_csv("GCDC_Dataset/yelp_train.csv", index_col=0)
enron   = pd.read_csv("GCDC_Dataset/enron_train.csv", index_col=0)

clinton['sample_weight'] = clinton.apply(lambda row: len(clinton[clinton.expert_label==row.expert_label])/len(clinton), axis=1)
yahoo['sample_weight'] = yahoo.apply(lambda row: len(yahoo[yahoo.expert_label==row.expert_label])/len(yahoo), axis=1)
yelp['sample_weight'] = yelp.apply(lambda row: len(yelp[yelp.expert_label==row.expert_label])/len(yelp), axis=1)
enron['sample_weight'] = enron.apply(lambda row: len(enron[enron.expert_label==row.expert_label])/len(enron), axis=1)

clinton_few_shot_idxs = dict()
yahoo_few_shot_idxs = dict()
yelp_few_shot_idxs = dict()
enron_few_shot_idxs = dict()

# Sample k% of rows, weighted by class label
for k in few_shot_percentages:
    clinton_few_shot_idxs[k] = clinton.sample(frac=k, weights='sample_weight', random_state=42).index.tolist()
    yahoo_few_shot_idxs[k] = yahoo.sample(frac=k, weights='sample_weight', random_state=42).index.tolist()
    yelp_few_shot_idxs[k] = yelp.sample(frac=k, weights='sample_weight', random_state=42).index.tolist()
    enron_few_shot_idxs[k] = enron.sample(frac=k, weights='sample_weight', random_state=42).index.tolist()

json.dump(clinton_few_shot_idxs, open("GCDC_Dataset/clinton_few_shot_idxs.json", 'w'))
json.dump(yahoo_few_shot_idxs, open("GCDC_Dataset/yahoo_few_shot_idxs.json", 'w'))
json.dump(yelp_few_shot_idxs, open("GCDC_Dataset/yelp_few_shot_idxs.json", 'w'))
json.dump(enron_few_shot_idxs, open("GCDC_Dataset/enron_few_shot_idxs.json", 'w'))


## WSJ

wsj_sent_jumble = pd.read_csv("WSJ_Dataset/train_sentence_jumbling.csv", index_col=0)
wsj_para_jumble = pd.read_csv("WSJ_Dataset/train_para_jumbling.csv", index_col=0)

# As WSJ dataset contains doc and its ~20 permutations in an ordered manner,, simply store the first k% of idxs
wsj_sent_jumble_few_shot_idxs = dict()
wsj_para_jumble_few_shot_idxs = dict()

for k in few_shot_percentages:
    wsj_sent_jumble_few_shot_idxs[k] = list(range(ceil(k*len(wsj_sent_jumble))))
    wsj_para_jumble_few_shot_idxs[k] = list(range(ceil(k*len(wsj_para_jumble))))

json.dump(wsj_sent_jumble_few_shot_idxs, open("WSJ_Dataset/train_sentence_jumbling_few_shot_idxs.json", 'w'))
json.dump(wsj_para_jumble_few_shot_idxs, open("WSJ_Dataset/train_para_jumbling_few_shot_idxs.json", 'w'))
