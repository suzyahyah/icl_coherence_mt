#!/usr/bin/env bash
# Author: Suzanna Sia

logs_e=$(pwd)/logs_e
logs_o=$(pwd)/logs_o

######### loop settings

DIRS=(en-fr) #{en-fr,en-de,en-pt}
MODELS=(bloom7b1) # xglm2.9B bloom3b)
#MODELS=(xglm2.9B) #{gptn2.7B,xglm2.9B,bloom3b,bloom7b1}
MODELS=(gptn2.7B) #uncomment for testing
GPU=v100 #{rtx,v100}

DOCLEVEL=0 #; sampling from outside document 
EXPS=(random nn_nob bm25_nob submodopt_nob)

#DOCLEVEL=1 #; sampling from within document for various settings
#EXPS=(window nn_otf_nob nn_otf_windowb bm25_otf_nob bm25_otf_windowb submodopt_otf_nob submodopt_otf_windowb shuffle rnd_unord static)

QSUB=1 #{-1,0,1}

# window and sim search prompt experiments are deterministic, no point running different seeds
for seed in 0; do 
for model in ${MODELS[@]}; do 
for direction in ${DIRS[@]}; do
for exp in ${EXPS[@]}; do
    
    if [[ $DOCLEVEL == 1 ]]; then
        exp=doclevel/${exp}
    fi

    cf=configs/prompt_select/${exp}.yaml
    [ ! -f "$cf" ] && echo "config file $cf does not exist" && exit 1

    args="$seed $model $direction $cf"
    echo $args

    if [[ $QSUB == 0 ]]; then
        bash bin/submit_docmt.sh $args
    elif [[ $QSUB == 1 ]]; then
        qsubname=${model}_${cf_}${direction}
        settings="-l mem_free=50G,gpu=1,h_rt=30:00:00 -q gpu.q@@$GPU"
        qsub -N $qsubname $settings -e $logs_e -o $logs_o bin/submit_docmt.sh $args
    else
        echo "Do nothing"
    fi

done
done
done
done
