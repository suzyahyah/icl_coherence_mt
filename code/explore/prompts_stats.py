#!/usr/bin/python3
# Author: Suzanna Sia

### Third Party imports
import pathlib
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader 
from sentence_transformers import SentenceTransformer


### Local/Custom imports
from code.datasets.utils import tokenize_text

rd = lambda x: np.around(x, 2)

def nn_sim_analysis(args, cfp, prompt_ds):
    test_df = prompt_ds.ds2.df
    sent_embed_model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    all_nn_dist = []

    for i in tqdm(range(len(test_df))):
        query = test_df.iloc[i]['source']
        prompts = prompt_ds.get_vals(i, query)
        query_embed = sent_embed_model.encode(query)
        prompt_embed = sent_embed_model.encode(prompts['source'].values)
        nn_dist = np.linalg.norm(query_embed - prompt_embed, axis=1).mean()
        all_nn_dist.append(rd(nn_dist))

    fn = cfp['nn_sim_analysis'].format(**(args))
    pathlib.Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)

    np.savetxt(fn, np.array(all_nn_dist))
    print("saved word analysis to:", fn)



def word_overlap_analysis(args, cfp, prompt_ds):
    # this is used in main function
    test_df = prompt_ds.ds2.df
    all_stats = []

    for i in tqdm(range(len(test_df))):

        query = test_df.iloc[i]['source']
        prompts = prompt_ds.get_vals(i, query)

        query_tok = tokenize_text(query)
        prompts_src_tok = tokenize_text(" ".join(prompts['source'].values))
        prompts_trg_tok = tokenize_text(" ".join(prompts['target'].values))

        stats = {}

        # here we compute the overlap
        src_overlap = set(query_tok).intersection(set(prompts_src_tok))
        len_src_overlap = len([w for w in query_tok if w in src_overlap])
        #len_src_overlap = len(src_overlap)
        if len_src_overlap == 0:
            stats['len_src_overlap'] = 0
            stats['R1_recall'] = 0
            stats['R1_precision'] = 0
            stats['R1_F1'] = 0
        else:

            R1_recall = len_src_overlap / len(prompts_src_tok)
            R1_precision = len_src_overlap / len(query_tok)

            stats['len_src_overlap'] = len_src_overlap
            stats['R1_recall'] = R1_recall
            stats['R1_precision'] = R1_precision
            stats['R1_F1'] = 2 * (R1_recall * R1_precision) / (R1_recall + R1_precision)

        # this is roughly equivalent to:
        # scores = rouge.get_scores(" ".join(query_tok), " ".join(prompts['source'].values))

        per_src_lens = [len(tokenize_text(s)) for s in prompts['source'].values]
        per_trg_lens = [len(tokenize_text(s)) for s in prompts['target'].values]

        per_src_len_mean = np.mean(per_src_lens)
        per_trg_len_mean = np.mean(per_trg_lens)


        stats['total_prompt_budget'] = len(prompts_src_tok) + len(prompts_trg_tok)
        stats['per_prompt_budget'] = per_src_len_mean + per_trg_len_mean
        stats['per_src_len'] = per_src_len_mean
        stats['per_trg_len'] = per_trg_len_mean
        stats['query_len'] = len(query_tok)

        stats['nprompts_used'] = len(prompts)
        all_stats.append(stats)

    fn = cfp['word_analysis'].format(**(args))

    pathlib.Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    all_stats = pd.DataFrame(all_stats).apply(rd)

    all_stats.to_csv(fn)
    print("saved word analysis to:", fn)


@torch.no_grad()
def perplexity_analysis(args, cfp, model, tokenizer, dataloader):

    all_logits = []
    # do we care about different measures of perplexity here?
    # conditional all prompts, source ppl
    # conditional - zeroshot source ppl etc.

    for j, batch in enumerate(tqdm(dataloader)):
        print(j, end=" ")
        # we need to do this one by one because there is no way to get individual losses
        query_lens = batch['query_len'][0]
        #max_len = batch['input_ids'].shape[1] + batch['query_ids'].shape[1]*2
        #max_len = min(2045, max_len)
        labels = batch['input_ids'].clone().cuda()
        labels[0][:-query_lens] = -100
        #for i, tok_ids in enumerate(labels):
        #    tok_ids[:-query_lens[i]] = -100

        with torch.no_grad():
            outputs = model(batch['input_ids'], labels=labels)

        all_logits.append(outputs.loss.detach().cpu().numpy())
        
    #save_fn = cfp['ppl_input_analysis'].format(**(args))
    save_fn = cfp['ppl_test_analysis'].format(**(args))
    pathlib.Path(os.path.dirname(save_fn)).mkdir(parents=True, exist_ok=True)
    
    np.save(save_fn, all_logits)
    print("saved ppl for source to:", save_fn)

def main(args, cfp, model, tokenizer, collate_fn, prompt_ds):
    word_overlap_analysis(args, cfp, prompt_ds) 
    nn_sim_analysis(args, cfp, prompt_ds)

    dataloader_ = DataLoader(prompt_ds,
                             collate_fn=collate_fn,
                             batch_size=1)

    perplexity_analysis(args,cfp, model, tokenizer, dataloader_)

