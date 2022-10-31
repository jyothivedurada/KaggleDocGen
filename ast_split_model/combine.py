import os
import sys

def combine_predictions(predictions):
    combined_prediction = []
    step_number = 1
    for prediction in predictions:
        combined_prediction.append("STEP " + str(step_number) + ": " + prediction)
        step_number += 1
    return combined_prediction

if __name__ == "__main__":
    pass