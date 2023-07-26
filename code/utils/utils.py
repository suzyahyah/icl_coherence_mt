#!/usr/bin/python3
# Author: Suzanna Sia

# Standard imports
#from code.decoder_hacks.stopping_criteria import QStoppingCriteria
import pandas as pd

import torch
from tqdm import tqdm
#from transformers import StoppingCriteriaList


@torch.no_grad()
def gen_text(dc_cfg, format_cf, model, tokenizer, dataloader, logitsp, args=None):

    all_gen_text = []
    all_ids = []

    for j, batch in enumerate(tqdm(dataloader)):
        all_ids.append(batch['ids'])
        # the max position embeddings for gptneo is 2048
        max_cutoff = dc_cfg['window_len'] - dc_cfg['gen_tokens_len']

        if batch['input_ids'].shape[1] > max_cutoff:
            print(f"max cutoff at batch {j}")
            batch['input_ids'] = batch['input_ids'][:, -max_cutoff:]
            batch['input_mask'] = batch['input_mask'][:, -max_cutoff:]

        max_len = batch['input_ids'].shape[1] + batch['query_ids'].shape[1]*2 # longer for cn
        max_len = min(2045, max_len)

        sep_ = format_cf['sep']
        l1_delim = format_cf['L1_delim']['value']
        l2_delim = format_cf['L2_delim']['value']

        #stopc = QStoppingCriteria(qid, len(batch['input_ids'][0]))
        #stopping_criteria = StoppingCriteriaList([stopc])

        trainable_prefix = tokenizer.additional_special_tokens
        bad_words_ids = None
        if len(trainable_prefix) > 0:
            bad_words_ids = tokenizer.batch_encode_plus(trainable_prefix)['input_ids']
            if "XGLM" in type(model).__name__:
                bad_words_ids = [id[1:] for id in bad_words_ids]

        # this code should be moved to the batch construction code
        model = build_causal_mask(args, model, batch, tokenizer)

        outputs = model.generate(batch['input_ids'],
                                 bad_words_ids=bad_words_ids,
                                 use_cache=True,
                                 attention_mask=batch['input_mask'],
                                 logits_processor=logitsp,
                                 max_length=max_len,
                                 pad_token_id=tokenizer.pad_token_id,
                                 do_sample=False,
                                 return_dict_in_generate=True,
                                 output_scores=True)

        gen_ids = outputs.sequences

        # later versions of huggingface dont need to find start_ix
        start_ix = batch['input_ids'].shape[1]
        gen_text = tokenizer.batch_decode(gen_ids[:, start_ix:])
        gen_text = [t[:t.find(sep_)+1] if sep_ in t else t for t in gen_text]

        if l1_delim.strip() != "":
            gen_text = [t.split(l1_delim)[0].strip().replace(sep_, "") for t in gen_text]

        if type(l2_delim) != str:
            l2_delim = l2_delim[0]

        if l2_delim.strip() != "":
            gen_text = [t.split(l2_delim)[0].strip().replace(sep_, "") for t in gen_text]

        # we always need to remove newline carriage otherwise the generated text file will
        # screw up badly. 
        gen_text = [t.replace("\n", "") for t in gen_text]
        gen_text = [t.replace("</s>", "") for t in gen_text]
        gen_text = [t.replace("<pad>", "") for t in gen_text]

        if j == 0:
            print(tokenizer.decode(batch['input_ids'][0]))
            print("\n====")
            print(gen_text)

        for i in range(len(gen_text)):
            all_gen_text.append({"id": batch['ids'][i], "gen_text": gen_text[i]})

    return all_gen_text 


def process_decode_mode(cfg, decode_mode):
    splits = decode_mode.split('_')

    if "beam" in decode_mode: 
        cfg['num_beams'] = int(splits[splits.index('beam')+1])

    if "repetitionp" in decode_mode:
        cfg['repetition_penalty'] = float(splits[splits.index('repetitionp')+1])

    if "sample" in decode_mode:
        cfg['do_sample'] = True 

    if "topp" in decode_mode:
        cfg['top_p'] = float(splits[splits.index('topp')+1])
    return cfg


def get_lang_from_langcodes(lang, lang_dict):
    if len(lang) > 2:
        key = "FLORES101-code"
    else:
        key = "MM100-code"
    lang = lang_dict[lang_dict[key]==lang]['language'].values[0]
    return lang


def set_lang_delim_tokens(decode_configs, direction, model_size): #, prefix):
    # either set as "English" "French" or "[0]" for special prefix

    lang_dict = pd.read_csv("assets/flores_map.csv", sep="\t")
    L1, L2 = direction.split('-')

    L1 = get_lang_from_langcodes(L1, lang_dict)
    L2 = get_lang_from_langcodes(L2, lang_dict)

    decode_configs['header'] = decode_configs['header'].replace("<L1>", L1)
    decode_configs['header'] = decode_configs['header'].replace("<L2>", L2)

    decode_configs['L1_delim']['value'] = decode_configs.L1_delim.value.replace("<L1>", L1)

    # we only modify L2
    if decode_configs.L2_delim.type == "string":
        decode_configs['L2_delim']['value'] = decode_configs.L2_delim.value.replace("<L2>", L2)

    elif decode_configs.L2_delim.type == "prefix":
        #decode_configs['L2_delim']['value'] = "".join(prefix_tokens) + " "
        prefix_tokens = [f'[[{i}]]' for i in range(decode_configs.L2_delim.n_toks)]
        prefix = "".join(prefix_tokens).strip()
        decode_configs['L2_delim']['value'] = decode_configs.L2_delim.value.replace("<L2>", prefix)
        #= "".join(prefix_tokens) + " "

    elif decode_configs.L2_delim.type == "prefix_surround":
        # surround the word French with special separators
        value = decode_configs.L2_delim.value.replace("<L2>", L2)
        if decode_configs.L2_delim.n_toks == 2:
            value = prefix_tokens[0] + value + prefix_tokens[1]
        elif decode_configs.L2_delim.n_toks == 1:
            value = prefix_tokens[0] + value
        else:
            raise Exception("not defined")
        decode_configs['L2_delim']['value'] = value
    else:
        raise Exception("not recognised delim type")

    if "xglm" in model_size:
        decode_configs['sep'] = "</s>"

    print(f"L1_delim:", decode_configs[f'L1_delim']) 
    print(f"L2_delim:", decode_configs[f'L2_delim']) 
    return decode_configs



def set_weights(model, prefix_weights, n_toks):
    if "XGLM" in type(model).__name__: 
        model.model.embed_tokens.weight.data[-n_toks:] = prefix_weights
    else:
        model.transformer.wte.weight.data[-n_toks:] = prefix_weights
    return model



def build_causal_mask(args, model, batch, tokenizer):
    if not args.model.hack:
        return model

    if "causal_mask" in args.model:
        # layer wise masking from experiments need to be batch 1
        # assert len(batch['ids']) == 1

        mask_instr = args.model.causal_mask.instructions
        mask_prompts = args.model.causal_mask.prompts

        batch_mask_from = []
        batch_mask_till = []
        
        for item in range(len(batch['input_ids'])):
            start = batch['input_ids'].shape[1] - batch['input_len'][0]
            mask_from, mask_till = None, None
            # mask_till = start

            if mask_instr:
                mask_from = start
                mask_till = start + batch['instructions_len'][item]

            if mask_prompts:
                if mask_from is None:
                    mask_from = start + batch['instructions_len'][item]
                else:
                    # use the mask from of instructions
                    pass
                
                mask_till = start + batch['prompt_len'][item] + 1

            batch_mask_from.append(mask_from)
            batch_mask_till.append(mask_till)

        if "Bloom" in str(model.__class__):
            num_layers = model.config.num_hidden_layers
        elif "GPT" in str(model.__class__):
            num_layers = model.config.num_layers


        for layer in range(args.model.mask_row, num_layers):

            if "Bloom" in str(model.__class__):
                model.transformer.h[layer].self_attention.mask_prev_positions = True
                model.transformer.h[layer].self_attention.mask_from = batch_mask_from
                model.transformer.h[layer].self_attention.mask_till = batch_mask_till
            elif "GPT" in str(model.__class__):
                model.transformer.h[layer].attn.attention.mask_prev_positions = True
                model.transformer.h[layer].attn.attention.mask_from = batch_mask_from
                model.transformer.h[layer].attn.attention.mask_till = batch_mask_till
            else:
                raise Exception("not implemented for model")
                # allow the model to see previous generated tokens, but not anything else
        #        input_edge = batch['input_ids'].shape[1]
        #        start_edge = [input_edge - batch['input_len'][i] for i in range(4)]
        #        model.transformer.h[layer].attn.attention.instr_start = start_edge
        #        model.transformer.h[layer].attn.attention.mask_till_ = [input_edge -x for x in batch['query_len']]

    return model


def get_default_argparser():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', default=0)
    argparser.add_argument('--data_cf', default='configs/data/default.yaml')
    argparser.add_argument('--prompt_select_cf',
                            default='configs/prompt_select/random_gptn.yaml')
    argparser.add_argument('--format_cf', default='configs/format/instr_L1L2.yaml')
    argparser.add_argument('--training_cf', default='configs/training/default.yaml')
    argparser.add_argument('--model_cf', default='configs/model/default.yaml')
    argparser.add_argument('--logitsp_cf', default='configs/logits_processor/default.yaml')
    argparser.add_argument('--generator_cf', default='configs/generator/default.yaml')
    argparser.add_argument('--file_paths_cfg', default="")

    return argparser

