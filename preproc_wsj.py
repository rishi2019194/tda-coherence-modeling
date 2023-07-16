'''
Read .jsonl files from WSJ dataset preprocessed by https://github.com/tushar117/Transformer-Models-for-Text-Coherence-Assessment,
Process them into permuted datasets in .csv format for both sentence jumbling and paragraph jumbling
'''
import pandas as pd
from tqdm import tqdm
from math import factorial
import random
import itertools

def prepare_df_for_processing(df):
    '''minor preparation before jumbling'''
    df = df.drop('facts', axis=1)
    df['length'] = [len(i) for i in df.sentences]
    return df

def sentence_jumbling(df):
    '''create upto 20 permutations of every doc and create a dataframe'''
    df_SJ = {'doc':[], 'expert_label':[]} # maintain similar column names as GCDC
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.length == 1: continue
        
        sent = ' '.join([' '.join(j) for j in row.sentences])

        df_SJ['doc'].append(sent)
        df_SJ['expert_label'].append(1) # label for coherent data sample
        
        n_permute = min(20, factorial(row.length))
        already_permuted = 0
        permuted_docs_set = set()
        permuted_docs_set.add(sent)

        while already_permuted < n_permute:
            permuted_doc = random.sample(row.sentences, row.length)
            permuted_doc = ' '.join([' '.join(j) for j in permuted_doc])
            
            if permuted_doc in permuted_docs_set: continue # duplicate

            df_SJ['doc'].append(permuted_doc)
            df_SJ['expert_label'].append(0)
            already_permuted+=1
    df_SJ = pd.DataFrame.from_dict(df_SJ)
    return df_SJ

def para_jumbling(df):
    '''create upto 20 permutations of every doc and create a dataframe. differs from sentence_jumbling in that only subgroups (paras) of the document are jumbled.'''
    df_PJ = {'doc':[], 'expert_label':[]} # maintain similar column names as GCDC
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.length < 20: continue # Only consider documents containing >5 paras of 4 sentences each i.e. 20 sentences
        
        sent = ' '.join([' '.join(j) for j in row.sentences])
        df_PJ['doc'].append(sent)
        df_PJ['expert_label'].append(1) # label for coherent data sample

        paras = [row.sentences[i:i+4] for i in range(0,row.length,4)] # split into paras of 4 sentences

        n_permute = min(20, factorial(len(paras)))
        already_permuted = 0
        permuted_docs_set = set()
        permuted_docs_set.add(sent)

        while already_permuted < n_permute:
            permuted_paras = random.sample(paras, len(paras))
            permuted_doc = list(itertools.chain.from_iterable(permuted_paras))
            permuted_doc = ' '.join([' '.join(j) for j in permuted_doc])
            
            if permuted_doc in permuted_docs_set: continue # duplicate

            df_PJ['doc'].append(permuted_doc)
            df_PJ['expert_label'].append(0)
            already_permuted+=1
    df_PJ = pd.DataFrame.from_dict(df_PJ)
    return df_PJ


train = pd.read_json(path_or_buf="WSJ_Dataset/train.jsonl", lines=True)
dev = pd.read_json(path_or_buf="WSJ_Dataset/dev.jsonl", lines=True)
test = pd.read_json(path_or_buf="WSJ_Dataset/test.jsonl", lines=True)

train = pd.concat([train,dev])

train = prepare_df_for_processing(train)
test = prepare_df_for_processing(test)

train_SJ = sentence_jumbling(train)
test_SJ = sentence_jumbling(test)

train_PJ = para_jumbling(train)
test_PJ = para_jumbling(test)

train_SJ.to_csv("WSJ_Dataset/sentence_jumbling_train.csv")
test_SJ.to_csv("WSJ_Dataset/sentence_jumbling_test.csv")

train_PJ.to_csv("WSJ_Dataset/para_jumbling_train.csv")
test_PJ.to_csv("WSJ_Dataset/para_jumbling_test.csv")
