#!/usr/bin/python3
# Author: Suzanna Sia

import os
import re
import pandas as pd
import pathlib
from code.datasets.data_utils import (filter_long_sentences, 
                                      lang_check_, remove_broke_lines,
                                      process_and_save_text, process_and_save_df, 
                                      add_line_id_to_doc_id)

mkpath = lambda x: pathlib.Path(os.path.dirname(x)).mkdir(parents=True, exist_ok=True)

def read_lines_strip(fn):
    with open(fn, 'r', encoding='utf-8') as f: 
        lines = f.readlines()
        lines = [l.strip() for l in lines] 
    return lines

def combine_files(mode, lang="fr", data_dir="data"):
    par_lines = []
    source = read_lines_strip(f'{data_dir}/TED/{mode}/en.txt.raw')
    target = read_lines_strip(f'{data_dir}/TED/{mode}/{lang}.txt.raw')

    for i, line in enumerate(target):
        par_lines.append("\t".join([str(i), source[i], target[i]]))

    par_lines.insert(0, "\t".join(['id', 'source', 'target']))
    with open(f'{data_dir}/TED/{mode}/en-{lang}.tsv', 'w') as f:
        f.write("\n".join(par_lines))

    print(f"written n lines to {data_dir}/TED/{mode}/en-{lang}.tsv:", len(par_lines))

def generate_doc_sep(mode, lang, data_dir, ntestdocs=120, ntestlines=120):

    mode_ = mode
    if mode == "test":
        mode_ = "test1"
    elif mode == "valid":
        mode_ = "dev"

    ddir = f"{data_dir}/TED/multitarget-ted"
    if mode == "train":
        talkid_file = f"{ddir}/en-{lang}/tok/ted_{mode_}_en-{lang}.tok.clean.seekvideo"
    else:
        talkid_file = f"{ddir}/en-{lang}/tok/ted_{mode_}_en-{lang}.tok.seekvideo"

    exp = r"<(.*?):"
    with open(talkid_file, 'r') as f:
        lines = f.readlines()
        
    doc_ids = [re.search(exp, lines[i]).group(1) for i in range(len(lines))]

    if mode == "train": # find lines that were removed  in clean
        import string

        trs = str.maketrans('', '', string.punctuation) 
        normal_fn = f'{ddir}/en-{lang}/tok/ted_train_en-{lang}.tok.en'
        clean_fn = f'{ddir}/en-{lang}/tok/ted_train_en-{lang}.tok.clean.en'

        with open(normal_fn, 'r') as f:
            normal_lines = f.readlines()
        with open(clean_fn, 'r') as f:
            clean_lines = f.readlines()

        sstrip = lambda x: re.sub(" +", " ", x.translate(trs).lower().strip())

        raw_lines = [sstrip(l) for l in normal_lines]
        clean_lines = [sstrip(l) for l in clean_lines]

        # find skipped lines
        i, j = 0, 0
        new_raw_lines, keep_lines = [], []
        while i < len(raw_lines):
            normal_toks = set(raw_lines[i].split())
            clean_toks = set(clean_lines[j].split())

            prop = len(normal_toks.intersection(clean_toks)) / len(normal_toks)
            if prop > 0.5:
                new_raw_lines.append(raw_lines[i])
                keep_lines.append(i)
                i += 1
                j += 1
            else:
    #            print("missing:", i, raw_lines[i], prop)
                i += 1
        
        assert len(clean_lines) == len(new_raw_lines)

    print(mode, len(set(doc_ids)))

    with open(f'{ddir}/en-{lang}/raw/ted_{mode_}_en-{lang}.raw.en', 'r') as f:
        en_data = f.readlines()
    with open(f'{ddir}/en-{lang}/raw/ted_{mode_}_en-{lang}.raw.{lang}', 'r') as f:
        lang_data = f.readlines()
    
    if mode == "train":
        en_data = [en_data[i] for i in keep_lines]
        lang_data = [lang_data[i] for i in keep_lines]

    en_data = [en.strip() for en in en_data]
    lang_data = [lang.strip() for lang in lang_data]

    df = pd.DataFrame()
    df['doc_id'] = doc_ids
    #df['line_id'] = line_ids
    df['source'] = en_data
    df['target'] = lang_data
    df = add_line_id_to_doc_id(df)
   
    if mode == "train":
        save_fn = f'data/TED/{mode}/en-{lang}.tsv.doc.clean'
    else:
        save_fn = f'data/TED/{mode}/en-{lang}.tsv.doc'

    df.to_csv(save_fn, sep="\t", index=False)
    print("saved to:", save_fn)

    # trainset  is TED-train-lt100
    # testset is TED-train-gt100
    if mode == "train":
        lengths_gt100, lengths_lt100 = [], []
        i = 0
        for k, grp in df.groupby('doc_id'):
            i += 1
        #    if i>ntestdocs:
        #        break

            if grp['line_id'].iloc[-1] < 100:
                lengths_lt100.append(grp)
            else:
                if len(lengths_gt100) > ntestdocs:
                    continue

                lengths_gt100.append(grp.iloc[:ntestlines])

        mkpath(f'data/TED/test-doc/en-{lang}.tsv.doc')
        mkpath(f'data/TED/train-doc/en-{lang}.tsv.doc')
        
        test_save_fn = f"data/TED/test-doc/en-{lang}.tsv.doc"
        train_save_fn = f"data/TED/train-doc/en-{lang}.tsv.doc"

        #lengths_gt100 = lengths_lt100[:ntestdocs]
        lengths_gt100 = pd.concat(lengths_gt100)
        lengths_lt100 = pd.concat(lengths_lt100)

        print("total unique doc ids:", len(df['doc_id'].unique()))
        print("ntest docs:", len(lengths_gt100['doc_id'].unique()))
        print("ntrain docs:", len(lengths_lt100['doc_id'].unique()))

        lengths_lt100.to_csv(train_save_fn, sep="\t", index=False)
        lengths_gt100.to_csv(test_save_fn, sep="\t", index=False)

    #process_and_save_df(df, save_fn, args.lang)
    #process_and_save_df_doc(df, save_fn, args.lang)


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--combine_docs', action="store_true", default=False)
    argparser.add_argument('--filter_lines', action="store_true", default=False)
    argparser.add_argument('--doc_sep', action="store_true", default=False)
    argparser.add_argument('--lang', default="fr")
    argparser.add_argument('--data_dir', default="data")

    args = argparser.parse_args()

    if args.combine_docs:
        for mode in ['train', 'test', 'valid']:
            combine_files(mode, lang=args.lang, data_dir=args.data_dir)

    if args.filter_lines:
        for mode in ['train', 'test', 'valid']:
            fn = f'{args.data_dir}/TED/{mode}/en-{args.lang}.tsv.raw'
            out_fn = f'{args.data_dir}/TED/{mode}/en-{args.lang}.tsv'
            remove_broke_lines(fn, out_fn)
            df = pd.read_csv(out_fn, names=['id', 'source', 'target'], header=0, delimiter="\t")
            save_fn = f'{args.data_dir}/TED/{mode}/en-{args.lang}.tsv'
            process_and_save_df(df, save_fn, args.lang)

    if args.doc_sep:
        # run this option if using on the fly translation 
        for mode in ['train']:
            generate_doc_sep(mode, lang=args.lang, data_dir=args.data_dir)

            
