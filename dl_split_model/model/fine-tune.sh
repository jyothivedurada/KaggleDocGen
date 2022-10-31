python3 /home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/model/run.py \
    --output_dir=/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/saved_models/by-ast-and-comments/epoch-2 \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=roberta-base \
    --do_train \
    --train_data_file=/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/dataset/splitted_data/by-ast-and-comments/positives-first/train.jsonl \
    --eval_data_file=/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/dataset/splitted_data/by-ast-and-comments/positives-first/valid.jsonl \
    --test_data_file=/home/cs19btech11056/cs21mtech12001-Tamal/Scripts/split_model/dataset/splitted_data/by-ast-and-comments/positives-first/test.jsonl \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee train.log