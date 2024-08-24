CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 few_shot_baseline_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/chatglm3-6b-32k \
    --test_path ../../data/CHEF-EG/test.json \
    --saved_dir ../../test_result \
    --save_name  5_shot_chatglm3_6b_32k.json \
    --seed 42 \
    --max_source_length 32768 \
    --max_target_length 300 \
    --model_type glm3
    
    