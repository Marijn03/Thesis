"""
Script for organizing and processing data from the SciTLDR-AIC dataset.

This script performs the following tasks:
1. Creates necessary directory structure for organizing files.
2. Reads data from the SciTLDR-AIC dataset in JSON Lines format.
3. Selects a random subset of data points.
4. Creates plain text files for datasets and prompts.
5. Creates empty files with ids for the summary of each model.
6. Saves the randomly selected papers of the SciTLDR-AIC dataset to a JSON file.

Functions:
- read_json(json_file): Read a JSON Lines file and convert it into a list of dictionaries.
- create_directory(directory, parent_dir): Create a directory if it does not exist.

"""

import json
import os
import random

def read_json(json_file):
    """ Read a JSON file, concatenate the 'source' texts from each JSON object into a single string,
    and estimate the summary length based on 25% of the number of the amount of sentences in dataset"""

    lst = []
    for count, line in enumerate(json_file):
        data = json.loads(line)
        concatenated_text = ' '.join(data['source'])
        nm = len(data['source']) * 0.25
        lst.append({"id":count, "source": concatenated_text, "summary_length": round(nm)})
    return lst

def create_directory(directory,parent_dir):
    """ Create a new directory at the specified location if it does not already exist. """
   
    try:
        path = os.path.join(parent_dir, directory) 
        os.mkdir(path)
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

def main():
    ################## DIRECTORY SETUP ##################
    # Creating articles_txt directory
    parent_dir = os.getcwd()
    article_directory = "article_txt"
    create_directory(article_directory,parent_dir)

    # Creating dataset directory in articles_txt directory
    create_directory("dataset", os.path.join(parent_dir, article_directory))

    # Creating dataset_prompt directory in articles_txt directory
    create_directory("dataset_prompt", os.path.join(parent_dir, article_directory))

    # Creating models directory
    model_directory = "models"
    create_directory(model_directory,parent_dir)

    # Creating chatgpt_3.5 directory in models directory
    create_directory("chatgpt_3.5", os.path.join(parent_dir, model_directory))

    # Creating cohere directory in models directory
    create_directory("cohere", os.path.join(parent_dir, model_directory))

    # Creating claude directory in models directory
    create_directory("claude", os.path.join(parent_dir, model_directory))

    ################## SELECT DATA FROM SciTLDR-AIC ##################
    # Specify the path to the JSON file
    file_path = os.path.join(parent_dir, 'scitldr-master/SciTLDR-Data/SciTLDR-AIC/train.jsonl') 
    with open(file_path, 'r') as json_file:
        paper_lst = read_json(json_file)

    random.seed(3)
    random_selection = random.sample(paper_lst, 19)

    ################## MAKE DATASET ################## 
    folder_path = os.path.join(parent_dir, "article_txt/dataset")
    folder_path_prompt = os.path.join(parent_dir, "article_txt/dataset_prompt")
    lst_paths_models = [os.path.join(parent_dir, "models/claude"), os.path.join(parent_dir, "models/chatgpt_3.5"), os.path.join(parent_dir, "models/cohere")]
    for paper in random_selection: 
        # Create dataset with plain text from SciTLDR-AIC dataset
        file_name = f"article_{paper['id']}.txt"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'w') as file:
            file.write(paper['source'])

        # Create dataset for prompting      
        prompt_1 = "What are the important entities in this document? What are the important dates in this document? What events are happening in this document? What is the result of these events? \nPlease answer the above questions:"         
        prompt_2 = f"You are a high school student who wants to gain more knowledge about the field of computer science or use this information in their high school report. Integrate the above information and make a short summary with max length of {paper['summary_length']} sentences of the article for a high school student."
        file_path_prompt = os.path.join(folder_path_prompt, file_name)
        with open(file_path_prompt, 'w') as file:
            file.write('{}\n\n{}\n\n{}\n'.format(paper['source'],prompt_1,prompt_2))

        # Create file structure for models with empty txt files        
        for model_path in lst_paths_models:
            file_path_model = os.path.join(model_path, file_name)
            with open(file_path_model, 'w') as file:
                file.write('\n')

    file_path = os.path.join(folder_path, "sentences.json")
    with open(file_path, 'w') as json_file:
        json.dump(random_selection, json_file, indent=2)


if __name__ == "__main__":
    main()