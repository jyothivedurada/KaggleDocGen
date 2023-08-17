lang=python #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=./notebooks-dataset/splitted_data/scscm
output_dir=./codoc/CodeBERT/output/notebooks_output
train_file=$data_dir/train_dataset.jsonl
dev_file=$data_dir/valid_dataset.jsonl
epochs=100
pretrained_model=microsoft/codebert-base #Roberta: roberta-base
load_model_path=./codoc/CodeBERT/output/csn-checkpoint-best-bleu/pytorch_model.bin

python3 ./codoc/CodeBERT/code/run.py \
        --do_train \
        --do_eval \
        --model_type roberta \
        --model_name_or_path $pretrained_model \
        --load_model_path $load_model_path \
        --train_filename $train_file \
        --dev_filename $dev_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --eval_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epochs