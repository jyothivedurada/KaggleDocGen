batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/notebooks_dataset/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/same_code_length
output_dir=/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_bart_summarization/same_code_length
dev_file=$data_dir/valid_dataset.jsonl
test_file=$data_dir/test_dataset.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python3 ./repository/Code-Text/code-to-text/code/run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size