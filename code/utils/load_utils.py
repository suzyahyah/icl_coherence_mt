#!/usr/bin/python3

import torch
import os
import pathlib
import pandas as pd
import pickle
import re
import json
from omegaconf import OmegaConf
from code.datasets.data_utils import get_fn_dataset
#os.environ['TRANSFORMERS_CACHE'] = "/brtx/605-nvme1/ssia1/.cache"
#os.environ['TRANSFORMERS_CACHE'] = "/exp/ssia/.cache"
from transformers.utils import WEIGHTS_NAME

def get_model_class(model_size, hack, args_model):
    #if "trl" in args_model:
        # transformer reinforcement learning library
    #    print("Loading from Transformer RL Library")
    #    from trl import AutoModelForCausalLMWithValueHead
    #    clsName = AutoModelForCausalLMWithValueHead
    #    return clsName

    if "gptn" in model_size:
        from transformers import GPTNeoForCausalLM
        clsName = GPTNeoForCausalLM
        if hack:
            from code.model_hacks import gptneo_model_hack
            clsName = gptneo_model_hack.GPTNeoForCausalLMHack

    if "bloom" in model_size:
        from transformers import AutoModelForCausalLM
        clsName = AutoModelForCausalLM
        if hack:
            from code.model_hacks import bloom_model_hack
            clsName = bloom_model_hack.BloomForCausalLMHack

    if "xglm" in model_size:

        from transformers import  XGLMForCausalLM
        clsName = XGLMForCausalLM
        if hack:
            raise Exception("xglm hack not implemented")

    print("Loading model class:", str(clsName.__class__))
    return clsName


def load_if_hack(clsName, og_fol, args_model):
    if args_model.hack:
        model = clsName.from_pretrained(og_fol, args_model)
        if save_fol != "":
            full_model_dict = model.state_dict()
            best_model_path = os.path.join(save_fol, WEIGHTS_NAME)
            masks_dict = torch.load(best_model_path)
            full_model_dict.update(masks_dict)
            model.load_state_dict(full_model_dict)
    else:
        model = clsName.from_pretrained(og_fol, torch_dtype=torch.bfloat16)
        #model = clsName.from_pretrained(og_fol, load_in_8bit=True)
    return model

def get_models(model_size="gptn2.7B", save_fol="", args_model=None, hack=False, cuda=True):
    #hack=False
    print(f"loading models from..{save_fol}")
    load_default = False
    clsName = get_model_class(model_size, args_model.layer_mask, args_model)

    if "gptn" in model_size:
        from transformers import GPT2Tokenizer
        size = model_size.replace("gptn", "")  
        og_fol = f"EleutherAI/gpt-neo-{size}"
        tokenizer = GPT2Tokenizer.from_pretrained(og_fol)
        # only for gptneo, not for xglm
        tokenizer.pad_token = tokenizer.eos_token
        model = load_if_hack(clsName, og_fol, args_model)

    elif "bloom" in model_size:
        from transformers import AutoTokenizer
        size = model_size.replace("bloom", "")  
        og_fol = f"bigscience/bloom-{size}"
        tokenizer = AutoTokenizer.from_pretrained(og_fol)
        model  = load_if_hack(clsName, og_fol, args_model) 

    elif "xglm" in model_size:
        from transformers import XGLMTokenizer, XGLMForCausalLM
        size = model_size.replace("xglm","")
        og_fol = f"facebook/xglm-{size}"
        tokenizer = XGLMTokenizer.from_pretrained(og_fol)
        tokenizer.bos_token_id = tokenizer.eos_token_id
        model  = load_if_hack(clsName, og_fol, args_model) 
        # because these geniuses initialised with the eos token instead

    elif "t5" in model_size:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        size = model_size.replace("t5","")
        save_fol = f"t5-{size}"
        tokenizer = T5Tokenizer.from_pretrained(save_fol)
        if not hack:
            model = T5ForConditionalGeneration.from_pretrained(save_fol)
    elif "opt" in model_size:
        from transformers import GPT2Tokenizer, OPTForCausalLM
        size = model_size.replace("opt", "")
        save_fol = f"facebook/opt-{size}"
        tokenizer = GPT2Tokenizer.from_pretrained(save_fol)
        if not hack:
            model = OPTForCausalLM.from_pretrained(save_fol)

    else:
        raise Exception("model not specified:", model_size)
        
    #else:
    # for batch decoding
    tokenizer.padding_side = "left"
    if cuda:
        model = model.cuda()
     #   model.to(torch.device("cuda"))
    print("loaded models..")
    return model, tokenizer


def gen_text_from_gpt(prompts, model, tokenizer, max_length=200):
    input_ids = tokenizer.batch_encode_plus(prompts, padding=True, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        max_length = len(input_ids[0]) + 40
        gen_tokens = model.generate(input_ids, max_length=max_length, pad_token_id=50256)
        gen_text = tokenizer.batch_decode(gen_tokens)
    return gen_text

def get_pastkv():
    output = model(input_ids, output_hidden_states=True, return_dict=True)
    output.past_key_values

def load_prefix_fn(args, cfg, cfp):
    args.lang = args.direction.split('-')[-1]
    args.n_toks = cfg.n_toks
    args.prefix_init = cfg.prefix_init
    args.join_tasks = cfg.join_tasks
    prefix_fn = cfp[''].format(**vars(args)) + f"/prefix.pt.s{args.prefix_seed}"
    return prefix_fn


def load_trained_prefix(model, tokenizer, args, cfg, cfp, cuda=True):
    if "prefix" not in args.decode_mode:
        return model, tokenizer, ""

    prefix_tokens = [f'[[{i}]]' for i in range(cfg.n_toks)]
    prefix = "".join(prefix_tokens).strip() 

    tokenizer.add_special_tokens({"additional_special_tokens": prefix_tokens})
    # prepare to load weights
    model.resize_token_embeddings(len(tokenizer))
    prefix_fn = load_prefix_fn(args, cfg, cfp) + f".s{args.prefix_seed}"
    prefix_weights = torch.load(prefix_fn)
    if cuda:
        prefix_weights.to(torch.device('cuda'))
        #prefix_weights = prefix_weights.cuda()
    print("loaded from:", prefix_fn)
    name = type(model).__name__

    if "XGLM" in name:
        model.model.embed_tokens.weight.data[-cfg.n_toks:] = prefix_weights
    else:
        model.transformer.wte.weight.data[-cfg.n_toks:] = prefix_weights

    #if args.train_type == "mono":
    if cuda:
        gen_ids = model.generate(tokenizer.encode(prefix, return_tensors='pt').cuda(), max_length=20)
        print("==sanity check the prefix token:")
        print(tokenizer.decode(gen_ids[0]))

    return model, tokenizer, prefix


def load_backtranslate(args, cfp, ds2, noprompt=False):
    args.direction = "-".join(args.direction.split("-")[::-1]) 
    # replace the previously generated model translations into the source 
    if noprompt:
        source_fn = cfp['gen_noprompt_fn'].format(**vars(args))
        source_fn = source_fn.replace("backtranslate", "noprompt")
    else:
        source_fn = cfp['gen_fn'].format(**vars(args))
        source_fn = source_fn.replace("backtranslate", "zeroshot")

    if not os.path.exists(source_fn):
        raise Exception("No file exists:", source_fn)
    with open(source_fn, 'r') as f: 
        lines = f.readlines()
    lines = [s.strip() for s in lines]
    ds2.df['source'] = lines
    #ds2.df.iloc[:len(lines)]['source'] = lines  
    args.direction = "-".join(args.direction.split("-")[::-1])

    return ds2



def load_val_set(args, decode_configs, tokenizer, raw=False):
    if args.val == "translation":
        from code.datasets.prompt_dataset import PromptsDataset 
        #val_dataset = load_translation_valset(args.domain, args.direction)
        # make sure we can generalise to the new prompt set and dev examples
        if args.domain == "wikipedia":
            if raw:
                mode = "dev_raw"
            else:
                mode = "dev"

            ds1 = get_fn_dataset("FLORES", mode, args.direction)
            ds2 = get_fn_dataset("FLORES", mode, args.direction)
        else:
            if raw:
                raise Exception("no code yet")
            ds1 = get_fn_dataset(args.domain, "valid", args.direction)
            ds2 = get_fn_dataset(args.domain, "valid", args.direction)
      
        # use this as val_dataset instead

        if "xglm" in args.model_size:
            nprompts = 3
        else:
            nprompts = 5
        if "2.9B" in args.model_size:
            bs = 2
        else:
            bs = 8

        val_dataset = PromptsDataset(decode_configs, ds1, ds2, 
                                    nprompts=nprompts, seed=0, sample_prefix=True,
                                    tokenizer=tokenizer,
                                    ntest=args.nval) #, batch_size=bs)
    else:
        from transformers import LineByLineTextDataset
        val_dataset = LineByLineTextDataset(tokenizer, file_path=valid_path, block_size=120)
    return val_dataset


def prepare_mono_data(args, cfp, tokenizer): 
    train_path = cfp[f'train_mono_fn'].format(**vars(args))
    #valid_path = cfp[f'valid_mono_fn'].format(**vars(args))
    block_size = 120

    train_path_prefix = process_prefix(args, train_path, tokenizer, block_size) 
    #valid_path_prefix = process_prefix(args, valid_path, tokenizer, block_size)
    print("train:", train_path_prefix)
    #print("valid:", valid_path_prefix)

    return train_path_prefix #, valid_path_prefix

def process_prefix(args, fn, tokenizer, block_size): #, lang, train_type="mono"):

    tokens = tokenizer.additional_special_tokens
    prefix_data_fn = f"{fn}.prefix.s{args.n_toks}"

    with open(fn, 'r') as f:
        data = f.readlines()

    print("processing prefix..")
    #if "ja" not in args.direction and "zh" not in args.direction:
    #    data = [s for s in data if len(s.split())>3 and len(s.split())<50]
    data = [s for s in data if not s.strip().isupper()]
    # remove everything inside parentheses 
    data = [re.sub('\((.*?)\)', '', line) for line in data]
    # substitute digites
    data = [re.sub('\d.?\d*', '_', line) for line in data]
    prefix = "".join(tokenizer.additional_special_tokens) # + ":"
    if "GPT" in tokenizer.__class__.__name__:
        data = [prefix+s.strip() for s in data][:args.mono_ntrain]
    elif "XGLM" in tokenizer.__class__.__name__:
        data = [prefix+s.strip()+tokenizer.eos_token for s in data][:args.mono_ntrain]

    with open(prefix_data_fn, "w") as f:
        f.write("\n".join(data))

    print("n lines:", len(data))
    print(data[0])
    print(data[1])

    return prefix_data_fn

