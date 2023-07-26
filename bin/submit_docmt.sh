#!/usr/bin/env bash
# Author: Suzanna Sia
#$ -l h=!r7n08*

cd /exp/ssia/projects/icl_coherence_mt
ml load cuda11.6/toolkit
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo $HOSTNAME $CUDA_VISIBLE_DEVICES
echo `nvcc --version`

seed=$1
model=$2
direction=$3
cf=$4

python code/run_prompts.py \
    --seed $seed \
    --model.model_size $model \
    --data.direction $direction\
    --prompt_select_cf $cf \
    --format_cf configs/format/instr_L1L2.yaml \
    --data.trainset TED \
    --data.testset TED \
    --data_cf configs/data/doc_level.yaml \
    --sample_prompts.nprompts 5 \
    --generator.batch_size 4 \
    --file_paths_cfg configs/file_paths/doc_mt.yaml
