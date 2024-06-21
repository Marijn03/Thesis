"""
Script to randomly select articles for the evaluation and developmen data from the final evaluation data. 
It saves it to JSON files.
"""
import os
import json
import random
from create_dataset import create_directory

# Load the final evaluation data from the JSON file
with open("evaluation/final_evaluation.json", "r") as eval_file:
    evaluation_data = json.load(eval_file)

# Randomly choose 5 articles
random.seed(3)
random_selection = random.sample(evaluation_data, 5)

# Create new directory annotated_dataset
parent_dir = os.getcwd()
create_directory("annotated_dataset", parent_dir)

# Write the randomly chosen articles to a file and put it in the directory annotated_dataset
with open('annotated_dataset/data_development.json', 'w') as json_file:
    json.dump(random_selection, json_file, indent=4)

evaluation_data = [x for x in  evaluation_data if x not in random_selection]
# Write the rest of the articles to a file and put it in the directory annotated_dataset 
with open('annotated_dataset/data_evaluation.json', 'w') as json_file:
    json.dump(evaluation_data, json_file, indent=4)