lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=/raid/cs21mtech12001/Research/CodeBERT/Repository/GraphCodeBERT/code-summarization/output/notebooks/with_spacy_summarization/all_constraints/todo-18
dev_file=/raid/cs21mtech12001/Research/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/valid_dataset.jsonl
test_file=/raid/cs21mtech12001/Research/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18/test_dataset.jsonl
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test
pretrained_model=microsoft/graphcodebert-base

python3 /raid/cs21mtech12001/Research/CodeBERT/Repository/GraphCodeBERT/code-summarization/run.py \
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