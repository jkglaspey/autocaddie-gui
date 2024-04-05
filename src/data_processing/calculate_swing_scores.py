import csv
import math
import os

# Calculate MAE
def calculate_mae(pred, act):
    if len(pred) != len(act):
        raise ValueError("Length of pred and act must be the same")
    
    n = len(pred)
    sum_absolute_error = sum(abs(pred[i] - act[i]) for i in range(n))
    return sum_absolute_error / n

# Convert MAE to a percentage rating between 0 and 100
def convert_to_percent(mae_val, max_possible_error):
    return 100 * (1 - (mae_val / max_possible_error))

# Read angles from a CSV file and return a list of angles
def read_angles(file_path):
    angles = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            angles.append(float(row[0]))  # Assuming the angle is in the first column
    return angles

# Find the score between two files containing angles
def find_score(file_path1, file_path2):
    pred_angles = read_angles(file_path1)
    act_angles = read_angles(file_path2)
    mae = calculate_mae(pred_angles, act_angles)
    max_possible_error = 180  # Maximum possible error for angles
    percent_score = convert_to_percent(mae, max_possible_error)
    return percent_score

def find_all_scores():

    # File paths
    file_path_pred_cam1 = r'neural_network\angle_csv\amateur_angles1.csv'
    file_path_pred_cam2 = r'neural_network\angle_csv\amateur_angles2.csv'
    file_path_act_cam1 = r'neural_network\amateur_swing_data1.csv'
    file_path_act_cam2 = r'neural_network\amateur_swing_data2.csv'

    # List of scores
    all_scores = []

    # Scores for posture
    # Insert code here

    # Scores for shoulder rotation
    scores = [find_score(file_path_pred_cam1, file_path_act_cam1), find_score(file_path_pred_cam2, file_path_act_cam2)]


    # Scores for forward arm

    # Scores for hip rotation
    
