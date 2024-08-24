CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 lora_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Qwen2-7B-Instruct \
    --lora_path ../../output/qwen2_lora_epoch4_trending \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir ../../test_result \
    --save_name  qwen2_lora_epoch4_trending.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type qwen2
    
    