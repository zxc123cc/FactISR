echo '开始训练'

device=$1
python train_classification_bert.py\
    --save_model_path ../../output/bert_base_trending \
    --device_ids 0 \
    --device 'cuda' \
    --num_workers 6 \
    --prefetch 12 \
    --pretrain_model_dir ~/dolphinfs_hdd_hadoop-dpsr/zhangxiaocheng/opensource_model_weights/tiansz/bert-base-chinese \
    --max_epochs 6 \
    --train_batch_size 12 \
    --val_batch_size 16 \
    --max_length 512 \
    --gradient_accumulation_steps 1 \
    --train_file ../../data/TrendFact/train.json \
    --dev_file ../../data/TrendFact/dev.json \
    --test_file ../../data/TrendFact/test.json \
    --seed 3407





