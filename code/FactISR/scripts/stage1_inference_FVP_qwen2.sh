LORA_DIR=../../output/Qwen2_stage1_train_ep5_lr3e-4
SAVE_DIR=../../test_result/stage1/Qwen2_stage1_train_ep5_lr3e-4

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage1_eval_fold.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ${LORA_DIR}/fold0 \
    --test_path ../../data/CHEF-EG/train.json \
    --saved_dir $SAVE_DIR \
    --save_name  FVP_fold0.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2 \
    --task_mode FVP \
    --fold 0
    

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage1_eval_fold.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ${LORA_DIR}/fold1 \
    --test_path ../../data/CHEF-EG/train.json \
    --saved_dir $SAVE_DIR \
    --save_name  FVP_fold1.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2 \
    --task_mode FVP \
    --fold 1
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage1_eval_fold.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ${LORA_DIR}/fold2 \
    --test_path ../../data/CHEF-EG/train.json \
    --saved_dir $SAVE_DIR \
    --save_name  FVP_fold2.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2 \
    --task_mode FVP \
    --fold 2
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage1_eval_fold.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ${LORA_DIR}/fold3 \
    --test_path ../../data/CHEF-EG/train.json \
    --saved_dir $SAVE_DIR \
    --save_name  FVP_fold3.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2 \
    --task_mode FVP \
    --fold 3
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage1_eval_fold.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ${LORA_DIR}/fold4 \
    --test_path ../../data/CHEF-EG/train.json \
    --saved_dir $SAVE_DIR \
    --save_name  FVP_fold4.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2 \
    --task_mode FVP \
    --fold 4