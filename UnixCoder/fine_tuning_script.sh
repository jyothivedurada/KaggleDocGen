lang=python
data_dir=/raid/cs21mtech12001/Research/Notebooks_Dataset/splitted_data/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18
output_dir=/raid/cs21mtech12001/Research/CodeBERT/Repository/UniXcoder/downstream-tasks/code-summarization/output/Notebooks/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/all_constraints/todo-18
trained_model=$output_dir/pytorch_model.bin

# Training
python3 /raid/cs21mtech12001/Research/CodeBERT/Repository/UniXcoder/downstream-tasks/code-summarization/run.py \
	--do_train \
	--do_eval \
	--model_name_or_path microsoft/unixcoder-base \
	--train_filename $data_dir/train_dataset.jsonl \
	--dev_filename $data_dir/valid_dataset.jsonl \
	--output_dir $output_dir \
	--max_source_length 256 \
	--max_target_length 128 \
	--beam_size 10 \
	--train_batch_size 48 \
	--eval_batch_size 48 \
	--learning_rate 5e-5 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 100 \
	--load_model_path $trained_model