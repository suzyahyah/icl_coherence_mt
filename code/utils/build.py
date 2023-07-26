#!/usr/bin/python3
# Author: Suzanna Sia
# pylint: disable=C0303,C0103

### Standard imports

### Custom imports
from code.datasets.data_utils import get_fn_dataset
from code.datasets.prompt_dataset import PromptsDataset      
from code.datasets.doclevel_prompt_dataset import DocLevelPromptsDataset      
from code.datasets.bm25_prompt_dataset import BM25DocLevelPromptsDataset, BM25PromptsDataset
from code.datasets.ppl_doclevel_prompt_dataset import PplDocLevelPromptsDataset
from code.datasets.nn_prompt_dataset import NNPromptsDataset, NNDocLevelPromptsDataset

from code.utils import load_utils

from code.datasets.data_utils import CollateFn

from transformers import TrainingArguments, EarlyStoppingCallback, Trainer


def build_model_tok(args, cfp, args_model, args_format):

    model, tokenizer = load_utils.get_models(args_model.model_size, 
                                             save_fol=args_model.save_fol,
                                             args_model=args_model,
                                             cuda=args_model.cuda,
                                             hack=args_model.layer_mask)

    return model, tokenizer


def build_datasets(args_data):

    ds_promptbank = get_fn_dataset(args_data.trainset, 'train', 
                                   args_data.direction, data_path=args_data.train_data_fn)

    ds_test = get_fn_dataset(args_data.testset, 'test',
                             args_data.direction, data_path=args_data.test_data_fn)

    
    return ds_promptbank, ds_test

def build_prompt_dataset(args, format_cf, model, tokenizer, ds_promptbank, ds_test):
    # Different classes correspond to different ways of sampling

    args_d = args.data
    args_p = args.sample_prompts

    if args.sample_prompts.sampling_method == "random":
        prompts_ds_class = PromptsDataset

    elif "bm25" in args.sample_prompts.sampling_method:
        if "doc_level" in args.sample_prompts.sampling_method:
            prompts_ds_class = BM25DocLevelPromptsDataset
        else:
            prompts_ds_class = BM25PromptsDataset

    elif "nn" in args.sample_prompts.sampling_method:
        if "doc_level" in args.sample_prompts.sampling_method:
            prompts_ds_class = NNDocLevelPromptsDataset
        else:
            prompts_ds_class = NNPromptsDataset

    elif "doc_level" in args.sample_prompts.sampling_method:
        if "ppl" in args.sample_prompts.sampling_method:
            prompts_ds_class = PplDocLevelPromptsDataset
        else:
            prompts_ds_class = DocLevelPromptsDataset

    prompt_ds = prompts_ds_class(format_cf, ds_promptbank, ds_test, 
                                 seed=args.seed,
                                 **args_p,
                                 ntest=args_d.ntest,
                                 model=model,
                                 tokenizer=tokenizer)
    return prompt_ds

