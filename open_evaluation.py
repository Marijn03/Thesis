"""
This script evaluates the open source summaries. It first puts the evaluation summaries (the summarie types) from the json file in directories, named with the article ID.
Then it loops through the text files to obtain the ranking for each knowledge setting (summary_type) and evaluation aspect. Then it concatinates the dataframes by 
grouping evaluation aspects and taking the mean.  
"""

from get_scores import extract_summary
from create_dataset import create_directory
import json 
import os
import pandas as pd

def extract_rankings(file_path):
    """Function to extract rankings from a text file"""

    # Initialize rankings with empty lists
    rankings = {
        '' : ['Language', 'Clarity and coherence', 'Detail', 'Confusing language'],
        'all_sum_ev': [0,0,0,0],
        'best_sum':[0,0,0,0],
        'zeroshot':[0,0,0,0],
        'all_sum':[0,0,0,0]
    }

    start_marker_lst = ["Language: low level of complex terms", "Clarity and coherence: clear structure or explanation of the concepts in the paper", "Detail: provide sufficient detail on the experimental results and significance of the findings", "Confusing language: vague language like “They fixed a thing called loss scaling”"]
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Process each line
    for line in content:
        line = line.replace("-", "").strip()  
        for start_marker in start_marker_lst:
            if start_marker in line:
                
                # Extract key
                start_index_key = line.find(":", 0)
                key = line[:start_index_key].strip()
                
                # Extract the summary text
                start_index_score = line.find("(", 0) + 1
                score = line[start_index_score:].replace(")", "").strip()

                # Split the string by commas to get individual rankings
                rankings_list = score.split(', ')

                # Process each ranking
                for ranking in rankings_list:
                    rank, model = ranking.split(':')
                    model = model.lower().strip()
                    rank = int(rank)
                    
                    if key.lower() == "language":
                        rankings[model][0] += rank
                    if key.lower() == "clarity and coherence":
                        rankings[model][1] += rank
                    if key.lower() == "detail":
                        rankings[model][2] += rank
                    if key.lower() == "confusing language":
                        rankings[model][3] += rank
    
    return pd.DataFrame(rankings)

def main():
    # Open json file with evaluation
    with open("result/evaluation_output.json", "r") as evaluation_output:
        output = json.load(evaluation_output)

    # Get the current directory and create a new directory if open_source_evaluation doesnt exist
    parent_dir = os.getcwd()
    create_directory("result/open_source_evaluation", parent_dir)

    # Put the summmary types from the json file in directories, named with the article ID.
    for article in output:
        for summary_type, summary in article['summaries'].items():
            folder_name = f"result/open_source_evaluation/{article['id']}"
            create_directory(folder_name, parent_dir)
            summary = extract_summary(summary)
            file_name = f"result/open_source_evaluation/{article['id']}/{summary_type}.txt"
            with open(file_name, "w") as file:
                file.write(f"{summary}\n")


    # Note: I have manually created the open_source_summaries_evaluation folder with the .txt files
    # Define open source evaluation folder_path
    parent_dir = os.getcwd()
    folder_path = os.path.join(parent_dir, "result/open_source_summaries_evaluation")

    # List all text files in the evaluation directory
    text_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Add the dataframes of the files in a list
    total_df = []
    for text_file in text_files:
        file_path = os.path.join(folder_path, text_file)
        df_ev = extract_rankings(file_path)
        total_df.append(df_ev)
    
    # Concatenate all DataFrames and group by evaluation_type
    sum_df = pd.concat(total_df).groupby('').mean().round(2)

    # Write the dataframe of ranking per aspect of the evulation and knowledge setting to a file
    sum_df.to_csv('result/open_evaluation_ranking.csv')

if __name__ == "__main__":
    main()