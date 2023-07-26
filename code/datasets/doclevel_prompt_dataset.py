#!/usr/bin/python3
# Author: Suzanna Sia
import pandas as pd
from code.datasets.prompt_dataset import PromptsDataset

class DocLevelPromptsDataset(PromptsDataset):
    # for document level prompts
    def __init__(self, decode_configs, ds1, ds2, nprompts=5, seed=0,
                 ntest=-1, shuffle_mode="", tokenizer=None, sample_on_new=False,
                 sampling_method="doc_level", filter_length=True, 
                 onthefly=True,
                 train_use_gold=True, test_use_gold=True, **kwargs):

        self.train_use_gold = train_use_gold
        self.test_use_gold = test_use_gold
        self.onthefly = onthefly

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

    def get_prefix(self, ds1, seed=0, i=0):
        vals = self.get_vals(i, query="")

        vals = vals.values
        vals = self._shufflevariants(vals)
        vals = [(v[0].strip(), v[1].strip()) for v in vals] 
        prefix = f"{self.sep}".join([f"{self.q}{v[0]}{self.a}{v[1]} " for v in vals])
        prefix = f"{self.header}{prefix}"
        return prefix

    def get_vals(self, i, query=""):
        # need to check the lineid number and docid number
        docid = self.ds2.df.iloc[i]['doc_id']
        lineid = self.ds2.df.iloc[i]['line_id']
        tkey1 = self.get_name_key("train")
        tkey2 = self.get_name_key("test")

        if self.filter_length:
            self.ds1.df_sample = self.ds1.df[self.ds1.df['long_enough']]
            ds2_s = self.ds2.df[self.ds2.df['long_enough']]
        else:
            self.ds1.df_sample = self.ds1.df
            ds2_s = self.ds2.df
            # unordered, we can sample anywhere in the document
            # except the current lineid
        if self.onthefly:
            #- ordered, we sample only before the line
            ds2_s = ds2_s[(ds2_s['line_id']<lineid) & (ds2_s['doc_id']==docid)]
        else:
            ds2_s = ds2_s[(ds2_s['doc_id']==docid) & (ds2_s['line_id']!=lineid)]
            vals = ds2_s.sample(n=self.nprompts)[[self.skey, tkey2]]
            return vals

        n_from_ds2 = min(len(ds2_s), self.nprompts)
        n_from_ds1 = self.nprompts - n_from_ds2

        if "static" in self.sampling_method:
            if n_from_ds2 == self.nprompts:
                # once we have our first 5 sentences, stop sampling from new
                self.sample_on_new = False

        #if int(lineid) < int(self.nprompts):
        # move this somewhere else

        if n_from_ds1 > 0:
            # we haven't translated enough of the document
            # this sample needs to be from stuff not in the document
            vals = self.ds1.df_sample.sample(n=n_from_ds1, random_state=self.seed)
            vals = vals[[self.skey, tkey1]]

            if n_from_ds2 > 0:
                vals2 = ds2_s.iloc[-n_from_ds2:][[self.skey, tkey2]]
                vals2 = vals2.rename(columns={tkey2: tkey1})
                # we gotta rename so it concats properly.
                vals = pd.concat([vals, vals2], axis=0)
        else:
            # use the translated document as prompts
            # we have to sanity check if the 
            # use the ordered generations
            #vals = self.ds2.df_sample.iloc[lineid-self.nprompts:lineid][[self.skey, self.tkey]]
            # this is correct, because we keep increasing the length of ds2_s 
            vals = ds2_s.iloc[-n_from_ds2:][[self.skey, tkey2]]
            # use the model generated sentences as prompts
        return vals
