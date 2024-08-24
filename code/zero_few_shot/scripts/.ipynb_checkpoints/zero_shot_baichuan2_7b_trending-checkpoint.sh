CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 zero_shot_baseline_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Baichuan2-7B-Chat \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir ../../test_result \
    --save_name  zero_shot_baichuan2_7b_chat_trending.json \
    --seed 42 \
    --max_source_length 4096 \
    --max_target_length 300 \
    --model_type baichuan2
    