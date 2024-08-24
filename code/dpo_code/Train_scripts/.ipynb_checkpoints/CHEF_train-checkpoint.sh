#!/bin/bash

# 运行 torchrun 命令
torchrun --nproc_per_node=2 /mnt/user/luyifei/LLaMA-Factory/src/train.py \
  --model_name_or_path /mnt/user/luyifei/model_weight/glm4_CHEF_lora_sft \
  --stage dpo \
  --do_train true \
  --finetuning_type lora \
  --lora_target all \
  --pref_beta 0.1 \
  --pref_loss sigmoid \
  --dataset CHEF_DPO_Train_data \
  --template glm4 \
  --cutoff_len 4096 \
  --max_samples 1000 \
  --overwrite_cache true \
  --preprocessing_num_workers 16 \
  --output_dir saves/glm4_CHEF_dpo \
  --logging_steps 10 \
  --save_steps 500 \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5.0e-6 \
  --num_train_epochs 2.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000 \
  --val_size 0.1 \
  --per_device_eval_batch_size 1 \
  --eval_strategy steps \
  --eval_steps 500
