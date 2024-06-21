import json
from collections import defaultdict
import pandas as pd
import os

def final_rank_ev(data):
    # Initialize a defaultdict to count ranks
    rank_counts = defaultdict(lambda: defaultdict(int))

    # Process each entry in the JSON
    for entry in data:
        for rank_info in entry['rank']:
            model_name = rank_info['model']
            rank_value = rank_info['rank']
            rank_counts[model_name][rank_value] += 1

    return rank_counts


def extract_rankings(file_path):
    """Function to extract rankings from a text file"""

    # Initialize rankings with empty lists
    rankings = {
        '' : ['Language', 'Clarity and coherence', 'Detail', 'Confusing language'],
        'chatgpt': [0,0,0,0],
        'claude':[0,0,0,0],
        'cohere':[0,0,0,0],
    }

    start_marker_lst = ["Language: low level of complex terms", "Clarity and coherence: clear structure or explanation of the concepts in the paper", "provide sufficient detail on the experimental results and significance of the findings", "Confusing language: vague language like “They fixed a thing called loss scaling”"]
    
    # Read the file content
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Process each line
    for line in content:
        line = line.replace("-", "").strip()  
        for start_marker in start_marker_lst:
            if start_marker in line:
                # print(line)
                
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
    # Define evaluation folder_path
    parent_dir = os.getcwd()
    folder_path = os.path.join(parent_dir, "evaluation")

    # List all text files in the evaluation directory
    text_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Makes list of the dataframes for each article
    total_df = []
    for text_file in text_files:
        file_path = os.path.join(folder_path, text_file)
        df_ev = extract_rankings(file_path)
        total_df.append(df_ev)

    # Concatenate all DataFrames and group by evaluation_type to sum ranks
    sum_df = pd.concat(total_df).groupby('').mean().round(2)

    # Write the dataframe of ranking per aspect of the evulation and model to a file
    sum_df.to_csv('result/closed_evaluation_ranking.csv')

    # Load JSON data from file
    with open('evaluation/final_evaluation.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # make dataframe of final rank for each article
    dct_final_rank = final_rank_ev(data)
    df = pd.DataFrame(dct_final_rank).T.fillna(0).astype(int)
    df = df.reindex(sorted(df.columns), axis=1) 
    df.to_csv('result/closed_final_rank.csv')

    # Make dictionary of the evaluation for each closed-source model and write it to a file 
    # (to easier manual evaluate closed-source models)
    
    dct_ev_str = defaultdict(str)
    for article in data:
        for key in article['rank']:
            dct_ev_str[key['model']] += key['evaluation'] + '\n'
    
    with open("result/manual_closed_ev.txt", "w") as file:
        for model in dct_ev_str:
            file.write(f"{model}\n")
            file.write(f"{dct_ev_str[model]}\n")

if __name__ == "__main__":
    main()
