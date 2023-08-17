lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=./codoc/GraphCodeBERT/output/notebooks/scscm
dev_file=./notebooks-dataset/splitted_data/scscm/valid_dataset.jsonl
test_file=./notebooks-dataset/splitted_data/scscm/test_dataset.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
pretrained_model=microsoft/graphcodebert-base

python3 ./codoc/GraphCodeBERT/run.py \
--do_test \
--model_type roberta \
--model_name_or_path $pretrained_model \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--load_model_path $load_model_path \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log