lang=python
data_dir=./notebooks-dataset/splitted_data/scscm
output_dir=./codoc/UniXcoder/output/Notebooks/scscm

# Evaluating
python3 ./codoc/UniXcoder/run.py \
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