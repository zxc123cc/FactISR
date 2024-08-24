#! /usr/bin/env bash
# pip3 install -r requirements.txt
set -ex

LR=3e-4
NUM_GPUS=4
LORA_RANK=128
LORA_ALPHA=256
LORA_DROUPOUT=0.05

MAX_SOURCE_LEN=16384 # longer context contains more paper titles, but cost longer train time and larger memory usage
MAX_TARGET_LEN=300
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=16
EPOCH=5
SAVE_INTERVAL=10
WARMUP_RATIO=0.03
SCHEDULAR=cosine

RUN_NAME=text
BASE_MODEL_PATH=/home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat
TRAIN_PATH=../../data/CHEF-EG/train.json
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=../../output/glm4_lora_epoch5
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


mkdir -p $OUTPUT_DIR

# deepspeed --include localhost:0 finetune.py\
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS lora_finetune.py \
    --train_format input-output \
    --train_data $TRAIN_PATH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROUPOUT \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers 10 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size  $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --num_train_epochs $EPOCH \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --bf16 \
    --deepspeed configs/deepspeed.json  2>&1 | tee ${OUTPUT_DIR}/train.log
