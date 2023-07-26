## In-context Learning as maintaining coherency: A study of On-the-fly Machine Translation using Large Language Models

Code and data setup for the paper: [pdf](https://arxiv.org/pdf/2305.03573.pdf)

### Getting Data

Run data preparation stages 0 to 4, modify `bin/data_prep/prepare-ted.sh` 

Stage 0: Get data
Stage 1: Combine monolingual files into bitext format
Stage 2: Filter bad lines
Stage 3: Reduce the promptbank (optinoal)
Stage 4: Get doc boundaries  

`bash bin/data_prep/prepare-ted.sh`

This repo currently focuses on the document level experiments.

### To Run Experiments:

#### Single experiment run
`bash bin/submit_docmt.sh $seed $model $direction $cf`

Example:
`bash bin/submit_docmt.sh 0 gptn125m en-fr configs/prompt_select/random.yaml`

#### Batch experiment run
`bash bin/batch_submit_docmt.sh` 

### Config files
Running different experiment settings comes from running different `configs/prompt_select/...yaml` files and `configs/prompt_select/doclevel/....yaml` files.

* `random`: random sample no budget constraints
* `bm25_nob`: bm25 no budget constraints
* `submodopt_nob`: submodular optimised selection no budget constraints
* `nn_nob`: Neural encoder similarity no budget constraints

`configs/prompt_select/doclevel/..`
* `window`: doclevel moving window

* `bm25_otf_nob`: BM25 similarity, onthefly, no budget 
* `bm25_otf_windowb`: BM25 similarity, onthefly, moving window's budget

* `submodopt_otf_nob`: Submodopt, onthefly no budget 
* `submodopt_otf_windowb`: Sumodopt, onthefly, moving window's budget

* `nn_otf_nob`: Neural encoder similarity, onthefly, no budget 
* `nn_otf_windowb`: Neural encoder similarity, onthefly, moving window's budget

Onthefly means prompts can only be sampled from previous translations and not anywhere in the document. 

* `shuffle`: sentences within moving window are shuffled
* `random_unord`: random within document
* `static`: Static first 5 translation pairs used as prompts throughout


