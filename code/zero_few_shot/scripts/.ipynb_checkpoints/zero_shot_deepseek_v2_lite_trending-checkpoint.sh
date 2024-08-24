CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 zero_shot_baseline_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/DeepSeek-V2-Lite-Chat \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir ../../test_result \
    --save_name  zero_shot_deepseek_v2_lite_trending.json \
    --seed 42 \
    --max_source_length 32768 \
    --max_target_length 300 \
    --model_type deepseek_v2
    
    