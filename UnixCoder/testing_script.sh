lang=python
data_dir=/raid/cs21mtech12001/Research/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/same_code_length
output_dir=/raid/cs21mtech12001/Research/CodeBERT/Repository/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/same_code_length

# Evaluating
python3 /raid/cs21mtech12001/Research/CodeBERT/Repository/UniXcoder/downstream-tasks/code-summarization/run.py \
	--do_test \
	--model_name_or_path microsoft/unixcoder-base \
	--test_filename $data_dir/test_dataset.jsonl \
	--output_dir $output_dir\
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 100 