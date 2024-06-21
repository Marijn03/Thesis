import os
import json

"""
This script reads evaluation data from the JSON file evaluation.json and assigns the summaries for each model from the directory 'models'. 
It also adds the original text and summary length from dataset/sentences.json to the corresponding ids in the evaluation data. 
It then writes the updated evaluation data to the JSON file 'final_evaluation.json'.

Example of format evaluation.json:
[{
        "id": 1125,
        "rank": {
            "1": {"model": "Chatgpt", "evaluation": "It contains a good explanation of CurricularFace and introduction of the use of CNN for face recognition, easy understandable language and mentioning one of the difference with traditional CL. However, it is  explicitly mentioning target audience and is missing a more detailed explanation of how CurricularFace is different from traditional models."},
            "2": {"model": "Claude", "evaluation": "It contains a good introduction of the topic, explaining difference with traditional CL and clear explanation how CurricularFace works. However, it is mentioning 'convergence issues' but does not explain this. Furthermore, the explanation of modulation function might be difficult to understand for high school students."},
            "3": {"model": "Cohere", "evaluation": "It is mentioning both differences between traditional CL and CurricularFace. However, the phrases 'discriminability of features' and 'flexible decision boundary' are difficult to understand. Furthermore 'through experiments on facial benchmarks' is not clearly explained in the summary. It is also missing structure. "}
        },
        "evaluation": "Chatgpt might miss some details in the explanation but explains everything very easily understandable. Claude uses easy language, but is not explaining some difficult concepts or explains concepts that might be difficult for high school students to understand. Cohere uses scientific writing which makes it difficult to understand, especially for high school students.",
        "chatgpt": null,
        "claude": null,
        "cohere": null
}]
"""

# Load the evaluation data from the JSON file
with open("evaluation/evaluation.json", "r") as eval_file:
    evaluation_data = json.load(eval_file)

# Load summaries for each model
chatgpt_summaries = {}
claude_summaries = {}
cohere_summaries = {}

# Function to read summaries from a directory
def read_summaries_from_directory(directory):
    summaries = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                summary = file.read()
                article_id = int(filename.split("_")[1].split(".")[0])
                summaries[article_id] = summary
    return summaries

# Read summaries for each model
chatgpt_summaries = read_summaries_from_directory("models/chatgpt_3.5")
claude_summaries = read_summaries_from_directory("models/claude")
cohere_summaries = read_summaries_from_directory("models/cohere")

# Read article text and add it to the item
with open("article_txt/dataset/sentences.json", "r") as sent_file:
    sent_data = json.load(sent_file)

# Assign summaries to evaluation data
for item in evaluation_data:
    item_id = item["id"]
    if item_id in chatgpt_summaries:
        item["chatgpt"] = chatgpt_summaries[item_id]
    if item_id in claude_summaries:
        item["claude"] = claude_summaries[item_id]
    if item_id in cohere_summaries:
        item["cohere"] = cohere_summaries[item_id]
    
    for article in sent_data:
        if article["id"] == item_id:
            item['article'] = article['source']
            item['summary_length'] = article['summary_length']

# Write the updated evaluation data to a new JSON file
with open("evaluation/final_evaluation.json", "w") as updated_eval_file:
    json.dump(evaluation_data, updated_eval_file, indent=4)