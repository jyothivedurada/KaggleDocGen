import nltk

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

chencherry = SmoothingFunction()
ground_truth_file = open('/home/cs19btech11056/cs21mtech12001-Tamal/HAConvGNN/repository/final_data/coms.test', 'r')
output_file = open('/home/cs19btech11056/cs21mtech12001-Tamal/HAConvGNN/repository/modelout/predictions/predict_notebook.txt', 'r')
ground_truth_file_lines = ground_truth_file.readlines()
output_file_lines = output_file.readlines()

print("\nNumber of lines in ground truth and output files are {} and {}".format(len(ground_truth_file_lines), len(output_file_lines)))

score = 0
for i in range(len(output_file_lines)):
    score += sentence_bleu([ground_truth_file_lines[i].split()], output_file_lines[i].split(), smoothing_function = chencherry.method1)
    
print("\nAverage BLUE score: ", score/len(output_file_lines))