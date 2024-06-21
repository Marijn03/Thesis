"""This script calculates the BERT F1-scores and extracts the summary from the text.
Then, it aggregates the matrix by summing the indexes of the 14 matrixes and divide it by the sum of all non-zero values. 
"""

from evaluate import load
bertscore = load("bertscore")
import json
import numpy

with open("result/evaluation_output.json", "r") as evaluation_output:
    output = json.load(evaluation_output)

with open("annotated_dataset/data_evaluation.json", "r") as file:
    data_evaluation = json.load(file)

def compute_bert(summary, ref):
    return bertscore.compute(predictions=[summary], references=[ref], model_type="distilbert-base-uncased")

def extract_summary(output):
    """Receive the summary from the text"""
    start_marker_lst = ["Here is", "Here's"]
    for start_marker in start_marker_lst:
        start_index = output.find(start_marker)
        if start_index != -1:
            start_index = output.find(":", start_index) + 1
            summary = output[start_index:].strip()
            return summary

references = []
for article in data_evaluation:
    dct_article = {}
    dct_article['id'] = article['id']
    dct_article['closed_summaries'] = {article[key['model'].lower()]:key['rank'] for key in article['rank']}
    dct_article['open_summaries'] = {summary_type: extract_summary(summary) for id in output if id['id'] == article['id'] for summary_type, summary in id['summaries'].items()}
    references.append(dict(dct_article))

# compute bert-score for each reference and summary by comparing the closed source summaries with each knowledge setting
all_matrixes = []
for article in references:
    matrix = [[0 for col in range(3)] for row in range(4)]
    i = 0
    for summary_type, open_summary in article['open_summaries'].items():
        rank_lst = []
        j=0
        for closed_summary, closed_rank in article['closed_summaries'].items():
            score = compute_bert(open_summary, closed_summary)
            if rank_lst and closed_rank == rank_lst[-1]:
                matrix[i][j-1] = (matrix[i][j-1] + score['f1'][0])/2
                matrix[i][j] = 0
            else:
                matrix[i][j] = score['f1'][0]
            rank_lst.append(closed_rank)
            j += 1
        i += 1
    all_matrixes.append(numpy.array(matrix))

# Initialize sum and count matrices
sum_matrix = numpy.zeros_like(all_matrixes[0], dtype=float)
count_matrix = numpy.zeros_like(all_matrixes[0], dtype=float)

# Sum values and count non-zero entries
for matrix in all_matrixes:
    sum_matrix += matrix
    count_matrix += (matrix != 0).astype(float)

result = sum_matrix/count_matrix

# Write the matrix to a file
numpy.savetxt("result/result_matrix.txt", result, fmt='%.7f')