# sh ./scripts/stage2_inference_FVP_glm4_trending.sh
# sh ./scripts/stage2_inference_EGP_glm4_trending.sh
# sh ./scripts/stage2_inference_SCP_glm4_trending.sh

SAVE_DIR=../../test_result/stage2/glm4_stage2_train_trending_ep4_rank64/beam1
NUM_BEAMS=1
LORA_PATH=../../output/glm4_stage2_train_trending_ep4_rank64

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  FVP_results.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode FVP \
    --num_beams $NUM_BEAMS





CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  EGP_results.json \
    --fact_label_path  ${SAVE_DIR}/FVP_results.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode EGP \
    --num_beams $NUM_BEAMS









CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  SCP_results_round1.json \
    --pseudo_explanation_path  ${SAVE_DIR}/EGP_results.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode SCP \
    --num_beams $NUM_BEAMS
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  SCP_results_round2.json \
    --pseudo_explanation_path  ${SAVE_DIR}/SCP_results_round1.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode SCP \
    --num_beams $NUM_BEAMS
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  SCP_results_round3.json \
    --pseudo_explanation_path  ${SAVE_DIR}/SCP_results_round2.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode SCP \
    --num_beams $NUM_BEAMS
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  SCP_results_round4.json \
    --pseudo_explanation_path  ${SAVE_DIR}/SCP_results_round3.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode SCP \
    --num_beams $NUM_BEAMS
    
    
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 stage2_inference.py \
    --model_path /home/hadoop-dpsr/dolphinfs_hdd_hadoop-dpsr/model_path/glm-4-9b-chat \
    --lora_path $LORA_PATH \
    --test_path ../../data/TrendFact/test.json \
    --saved_dir $SAVE_DIR \
    --save_name  SCP_results_round5.json \
    --pseudo_explanation_path  ${SAVE_DIR}/SCP_results_round4.json \
    --seed 42 \
    --max_source_length 16384 \
    --max_target_length 300 \
    --model_type glm4 \
    --task_mode SCP \
    --num_beams $NUM_BEAMS
    
    