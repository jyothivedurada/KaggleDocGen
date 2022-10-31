import os
import sys

def combine_predictions(predictions, indentations):
    combined_prediction = []
    step_number = 1
    for i in range(len(predictions)):
        if i == 0 or predictions[i] != predictions[i-1]:
            combined_prediction.append(" " * indentations[i] + "STEP " + str(step_number) + ": " + predictions[i])
        step_number += 1
    return combined_prediction

if __name__ == "__main__":
    pass