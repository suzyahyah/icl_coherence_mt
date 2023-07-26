#!/usr/bin/python3
# Author: Suzanna Sia
import numpy as np
import faiss
from code.datasets.prompt_dataset import PromptsDataset
from code.datasets.utils import tokenize_text
from sentence_transformers import SentenceTransformer


class NNPromptsDataset(PromptsDataset):

    def __init__(self, decode_configs, ds1, ds2, nprompts=5, seed=0,
                 ntest=-1, shuffle_mode="", tokenizer=None, sample_on_new=False,
                 sampling_method="doc_level", filter_length=True, do_submodopt=False,
                 onthefly=False,
                 train_use_gold=True, test_use_gold=True, per_p_budget=20, **kwargs):

        self.train_use_gold = train_use_gold
        self.test_use_gold = test_use_gold
        self.per_p_budget = per_p_budget
        self.onthefly = onthefly
        self.construct_index(ds1, ds2)
        self.do_submodopt = do_submodopt

        ds1.df['src_wc'] = ds1.df['source'].apply(lambda x: len(x.split()))
        ds1.df['target_wc'] = ds1.df['target'].apply(lambda x: len(x.split()))
        ds1.df['src_target_wc'] = ds1.df['src_wc'] + ds1.df['target_wc']

        ds2.df['src_wc'] = ds2.df['source'].apply(lambda x: len(x.split()))
        ds2.df['target_wc'] = ds2.df['target'].apply(lambda x: len(x.split()))
        ds2.df['src_target_wc'] = ds2.df['src_wc'] + ds2.df['target_wc']


        super().__init__(decode_configs, ds1, ds2, nprompts=nprompts, seed=seed,
                         ntest=ntest, shuffle_mode=shuffle_mode, tokenizer=tokenizer, 
                         sample_on_new=sample_on_new,
                         sampling_method=sampling_method, filter_length=filter_length)




    def get_name_key(self, mode):
        if mode == "train":
            use_gold = self.train_use_gold
        elif mode == "test":
            use_gold = self.test_use_gold

        if use_gold:
            tkey = self.tkey
        else:
            tkey = "target_gen"
        return tkey

    def construct_index(self, ds1, ds2):
        # the index is itself self.ds2
        # the query is also from ds2. 
        # construct the index from the current docid
        sent_embed_model = SentenceTransformer("all-mpnet-base-v2")
        ds1_vals = sent_embed_model.encode(ds1.df['source'].values)
        ds2_vals = sent_embed_model.encode(ds2.df['source'].values)
        self.index1 = faiss.IndexFlatL2(ds1_vals.shape[1])
        self.index1.add(ds1_vals)

        self.ds1_vals = ds1_vals
        self.ds2_vals = ds2_vals
        print("index constructed:", self.index1.is_trained, self.index1.ntotal)


    def get_prefix(self, ds1, seed=0, i=0):
        # need to check the lineid number and docid number
        query = self.ds2.df.iloc[i]['source']

        # use https://pypi.org/project/rank-bm25/
        if "static" in self.sampling_method:
            self.sample_on_new = False

        #if int(lineid) < int(self.nprompts):
        # move this somewhere else
        tkey1 = self.get_name_key("train")
        self.tkey2 = self.get_name_key("test")

        # get top docscores ilocs, sanity check them and then draw them out.
        # this can actually be :self.nprompts since source and target are not the sam
        vals = self.get_vals(i, query)


        vals = vals.values
        vals = self._shufflevariants(vals)
        vals = [(v[0].strip(), v[1].strip()) for v in vals] 
        prefix = f"{self.sep}".join([f"{self.q}{v[0]}{self.a}{v[1]} " for v in vals])
        prefix = f"{self.header}{prefix}"
        return prefix

    def get_total_budget(self, df, lineid):
        if self.per_p_budget == 0:
            # budget equivalent to sliding window
            total_budget = 0
            if lineid is None:
                total_budget = self.nprompts * 20
            else:
                total_budget = df.iloc[lineid-self.nprompts:lineid]['src_target_wc'].sum()

        elif self.per_p_budget == -1:
            #no budget
            total_budget = self.nprompts

        elif self.per_p_budget == -2:
            #rand budget 
            total_budget = df.sample(n=self.nprompts)['src_target_wc'].sum()

        elif self.per_p_budget > 0:
            total_budget = self.nprompts * self.per_p_budget

        return total_budget

    def get_budget(self, lineid, df):
        if self.per_p_budget == -1:
            budget = 1
        else:
            budget = df.iloc[lineid]['src_target_wc']

        return budget

    def allowable_doc_scores(self, doc_scores, rem_budget, df, excl=None, lineid=None):

        if excl:
            doc_scores[excl] = -999

        if lineid is not None:

            if lineid > self.nprompts:
                doc_scores[lineid] = -999
                if self.onthefly:
                    doc_scores[lineid:] = -999

        if self.per_p_budget < 0:
            pass

        else:
            df['oob'] = df['src_target_wc'].apply(lambda x: x > rem_budget)
            doc_scores[df['oob']] = -999

        return doc_scores, df
            

    def get_vals(self, i, query):
        ds = self.ds1.df
        index = self.index1
        vals = self.get_vals_from_ds_index(query, ds, i, index)

        return vals

    def get_vals_from_ds_index(self, query, df, i, index, lineid=None):
        query_embed = self.ds2_vals[i]
        budget = 0
        total_budget = self.get_total_budget(df, lineid)
        rem_budget = total_budget - budget
        nearest_scores = index.search(np.expand_dims(query_embed, 0), 100)

        # tuple of (distances, ids)
        ds = 10 - nearest_scores[0]
        doc_scores = np.zeros(index.ntotal)
        doc_scores[nearest_scores[1]] = ds

        selected_ids = []

        assert len(doc_scores) == len(df)

        while rem_budget > 0:
            doc_scores, df = self.allowable_doc_scores(doc_scores, 
                                                       rem_budget, df,
                                                       excl=selected_ids, lineid=lineid)

            if len(np.where(doc_scores < 0)[0]) == len(doc_scores):
                # if all docscores are negative, i.e. not valid
                break

            top_lineid = np.argsort(doc_scores)[::-1][0]

            budget += self.get_budget(top_lineid, df)
            rem_budget = total_budget - budget
            selected_ids.append(top_lineid)

        vals = df.iloc[selected_ids][[self.skey, self.tkey2]]
        return vals


class NNDocLevelPromptsDataset(NNPromptsDataset):
    # for document level prompts
    def __init__(self, decode_configs, ds1, ds2, nprompts=5, seed=0,
                 ntest=-1, shuffle_mode="", tokenizer=None, sample_on_new=False,
                 sampling_method="doc_level", filter_length=True, per_p_budget=20,
                 do_submodopt=False, train_use_gold=True, 
                 test_use_gold=True, onthefly=False, **kwargs):


        self.construct_index_doc(ds2)

        super().__init__(decode_configs, ds1, ds2, nprompts=nprompts, seed=seed,
                         ntest=ntest, shuffle_mode=shuffle_mode, tokenizer=tokenizer, 
                         sample_on_new=sample_on_new, per_p_budget=per_p_budget,
                         do_submodopt=do_submodopt, onthefly=onthefly,
                         sampling_method=sampling_method, filter_length=filter_length)


    def construct_index_doc(self, ds2):
        # the index is itself self.ds2
        # the query is also from ds2. 
        # construct the index from the current docid
        self.nn_indexes_doc = {}
        sent_embed_model = SentenceTransformer("all-mpnet-base-v2")

        for doc_id in ds2.df['doc_id'].unique():
            corpus = ds2.df[ds2.df['doc_id'] == doc_id]['source'].values
            ds2_vals = sent_embed_model.encode(corpus)
            self.nn_indexes_doc[doc_id] = faiss.IndexFlatL2(ds2_vals.shape[1])
            self.nn_indexes_doc[doc_id].add(ds2_vals)

        print("index constructed for ds2")

    def get_vals(self, i, query):

        docid = self.ds2.df.iloc[i]['doc_id']
        lineid = self.ds2.df.iloc[i]['line_id']
        #doc_scores = self.bm25_indexes_doc[docid].get_scores(tokenized_query)
        # available doc_scores
        # if onthefly, then we only consider sentences up to the current line.
        # it not onthefly, then we consider sentences anywhere in the doc
        # if corpus-combine, then we consider sentences from both training and document
        # if corpus-doclevel, then we consider sentences only from document
        # ifcorpus-none, then we consider sentences from training set
        #top_lineid = np.argsort(doc_scores)[::-1][1:self.nprompts+1]

        #if len(vals) < self.nprompts:
        if lineid > self.nprompts:
            ds2_s = self.ds2.df[self.ds2.df['doc_id'] == docid]
            index = self.nn_indexes_doc[docid]
            vals = self.get_vals_from_ds_index(query, ds2_s, i, index, lineid=lineid)
        else:
            df = self.ds1.df
            index = self.index1
            vals = self.get_vals_from_ds_index(query, df, i, index)

        return vals
