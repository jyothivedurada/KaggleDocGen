

output_file = open('/home/cs19btech11056/cs21mtech12001-Tamal/CodeXGLUE/output/notebooks_output/competition_notebooks_with_atleast_1_medal_and_10_votes/with_spacy_summarization/doc_length_15/test_1.output', 'r')
Lines = output_file.readlines()

no_of_lines = 0
total_words = 0
word_dictionary = {}
length_dictionary = {}
type = "codebert"
for line in Lines:
    no_of_lines += 1
    if type == "haconvgnn": 
        line = line.split("\t")[1].strip()
        try:
            line = line[:line.index('</s>') + 4].strip()
            line = line[4:-5].strip().lower()
        except:
            line = line[4:].strip().lower()
    else:
        line = " ".join(line.split("\t")[1:]).strip().lower()
    print("Line-{}: {}".format(no_of_lines, line))

    length = len(line.split(" "))
    total_words += length
    for word in line.split(" "):
        if word in word_dictionary:
            word_dictionary[word] += 1
        else:
            word_dictionary[word] = 1
    if length in length_dictionary:
        length_dictionary[length] += 1
    else:
        length_dictionary[length] = 1

    
print("\nAverage length of comments: {}".format(total_words/no_of_lines))
print("\n-----------------------------------------------------------------------------------------")
word_dictionary = {k: v for k, v in sorted(word_dictionary.items(), key=lambda item: item[1], reverse=True)}
length_dictionary = {k: v for k, v in sorted(length_dictionary.items(), key=lambda item: item[1], reverse=True)}
print(word_dictionary)
print("\n-----------------------------------------------------------------------------------------")
print("\n", length_dictionary)
output_file.close()