import pandas as pd
import re
import numpy as np

def process_cell(data):
    data = re.split(':|\n', data)
    correctness, informativeness, readability = int(data[1].strip()), int(data[3].strip()), int(data[5].strip())
    return correctness, informativeness, readability

HE_FILE_LOCATION = "/home/cs19btech11056/cs21mtech12001-Tamal/Results/HE2-Results.csv"

results_dict = {"CASE 1" : {"Correctness": [], "Informativeness": [], "Readability": []}, 
                "CASE 2" : {"Correctness": [], "Informativeness": [], "Readability": []}, 
                "CASE 3" : {"Correctness": [], "Informativeness": [], "Readability": []}, 
                "CASE 4" : {"Correctness": [], "Informativeness": [], "Readability": []}, 
                "CASE 5" : {"Correctness": [], "Informativeness": [], "Readability": []}, 
                "CASE 6" : {"Correctness": [], "Informativeness": [], "Readability": []}}

he_data = pd.read_csv(HE_FILE_LOCATION)
for index, row in he_data.iterrows():
    if str(row["CASE 1"]) != "nan" and \
       str(row["CASE 2"]) != "nan" and \
       str(row["CASE 3"]) != "nan" and \
       str(row["CASE 4"]) != "nan" and \
       str(row["CASE 5"]) != "nan" and \
       str(row["CASE 6"]) != "nan":
      for i in range(1, 7):
          case = "CASE " + str(i)
          correctness, informativeness, readability = process_cell(str(row[case]))
          results_dict[case]["Correctness"].append(correctness)
          results_dict[case]["Informativeness"].append(informativeness)
          results_dict[case]["Readability"].append(readability)
          
print(len(results_dict["CASE 1"]["Correctness"]))

for case in results_dict:
    for key in results_dict[case]:
        print("\nMean and Standard deviation for {} are {} and {}".format(case + " - " + key, np.mean(results_dict[case][key]), np.std(results_dict[case][key])))