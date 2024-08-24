CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 lora_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ../../output/lora_lora_epoch5 \
    --test_path ../../data/CHEF-EG/test.json \
    --saved_dir ../../test_result \
    --save_name  lora_qwen2_7b_beam1.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2
    
    