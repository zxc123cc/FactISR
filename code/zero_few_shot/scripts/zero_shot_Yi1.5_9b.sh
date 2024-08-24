CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 zero_shot_baseline_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/Yi-1.5-9B-Chat-16K \
    --test_path ../../data/CHEF-EG/test.json \
    --saved_dir ../../test_result \
    --save_name  zero_shot_yi_1.5_9b_chat_16k.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type Yi1_5
    