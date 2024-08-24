echo '开始训练'

device=$1
python train_classification_bert.py\
    --save_model_path ../../output/roberta_large \
    --device_ids 0 \
    --device 'cuda' \
    --num_workers 6 \
    --prefetch 12 \
    --pretrain_model_dir ~/dolphinfs_hdd_hadoop-dpsr/model_path/chinese-roberta-wwm-ext-large \
    --max_epochs 6 \
    --train_batch_size 12 \
    --val_batch_size 16 \
    --max_length 512 \
    --gradient_accumulation_steps 1 \
    --train_file ../../data/CHEF-EG/train.json \
    --dev_file ../../data/CHEF-EG/dev.json \
    --test_file ../../data/CHEF-EG/test.json \
    --seed 3407





