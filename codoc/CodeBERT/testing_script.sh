batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=./notebooks-dataset/splitted_data/scscm
output_dir=./codoc/CodeBERT/output/notebooks_output
dev_file=$data_dir/valid_dataset.jsonl
test_file=$data_dir/test_dataset.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python3 ./codoc/CodeBERT/code/run.py \
        --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $test_model \
        --dev_filename $dev_file \
        --test_filename $test_file \
        --output_dir $output_dir \
        --max_source_length $source_length \
        --max_target_length $target_length \
        --beam_size $beam_size \
        --eval_batch_size $batch_size