from transformers import AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
import transformers
import torch
import psutil
import json
import random
from evaluate import load
import os
from create_dataset import create_directory

bertscore = load("bertscore")
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

model = transformers.AutoModelForCausalLM.from_pretrained(
  model_id,
  trust_remote_code=True,
  torch_dtype=torch.bfloat16,
  low_cpu_mem_usage=True,
  cache_dir='/scratch/s4850971/caches'
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    cache_dir='/scratch/s4850971/caches')

pipeline = pipeline('text-generation', 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_new_tokens=8000, 
                    do_sample=True, 
                    use_cache=True, 
                    eos_token_id=tokenizer.eos_token_id, 
                    pad_token_id=tokenizer.eos_token_id)

llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.2})

def get_example_string_best(example):
    output_string = ""
    best_sum = ""
    output_string += f"Example article:\n"
    for model in example['rank']:
        if model['rank'] == 1:
            best_sum += f"{example[model['model'].lower()]}\n"

    output_string += f"Best summarie(s){best_sum}"

    return output_string

def get_example_string_sum(example):
    output_string = ""
    output_string += f"Example article:\n"
    output_string += f"{example['article']}\n\n"
    output_string += f"Chatgpt summary\n{example['chatgpt']}\n"
    output_string += f"Claude summary\n{example['claude']}\n"
    output_string += f"Cohere summary\n{example['cohere']}\n"
    overall_ranking = ""
    for rank in example['rank']:
        overall_ranking += f"{rank['model']}: {rank['rank']}\n"
    output_string += f"Overall Ranking\n{overall_ranking}\n"

    return output_string

def get_example_string_sum_ev(example):
    output_string = ""
    
    output_string += f"Example article:\n"
    output_string += f"{example['article']}\n\n"

    for rank in example['rank']:
        if rank['model'] == "Chatgpt":
            output_string += f"Chatgpt summary\n{example['chatgpt']}\n"
            output_string += f"Chatgpt evaluation\n{rank['evaluation']}\n\n"
        if rank['model'] == "Claude":
            output_string += f"Claude summary\n{example['claude']}\n"
            output_string += f"Claude evaluation\n{rank['evaluation']}\n\n"
        if rank['model'] == "Cohere":
            output_string += f"Cohere summary\n{example['cohere']}\n"
            output_string += f"Cohere evaluation\n{rank['evaluation']}\n\n"

    overall_ranking = ""
    for rank in example['rank']:
        overall_ranking += f"{rank['model']}: {rank['rank']}\n"
    output_string += f"Overall Ranking:\n{overall_ranking}\n"

    output_string += f"Overall evaluation:\n{example['evaluation']}\n"

    return output_string

def extract_summary(output):
    """Receive the summary from the text"""

    start_marker_lst = ["Here is a summary", "Here's a summary"]
    for start_marker in start_marker_lst:

        # Find the start of the summary
        start_index = output.find(start_marker)
        if start_index != -1:
            # Move the index to the end of the marker to get the actual summary content
            start_index = output.find(":", start_index) + 1
            
            # Extract the summary text
            summary = output[start_index:].strip()
            
            return summary

template_zero = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Please answer the following questions:
1. What are the important entities in this document? 
2. What are the important dates in this document? 
3. What events are happening in this document? 
4. What is the result of these events? 

Summarize the given scholarly document for a high school student in a way that embeds answers to the above questions within the summary itself.
Ensure the summary is coherent and does not explicitly list the questions and answers, with a maximum length of {summary_length} sentences.

Scholarly document
{new_article}
<|eot_id|>
<|start_header_id|>assistent<|end_header_id|>
"""

template_few = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{examples}
Please answer the following questions:
1. What are the important entities in this document? 
2. What are the important dates in this document? 
3. What events are happening in this document? 
4. What is the result of these events? 

Summarize the given scholarly document for a high school student in a way that embeds answers to the above questions within the summary itself.
Ensure the summary is coherent and does not explicitly list the questions and answers, with a maximum length of {summary_length} sentences.

Scholarly document
{new_article}
<|eot_id|>
<|start_header_id|>assistent<|end_header_id|>
"""

# prompt = PromptTemplate(template=template_few, input_variables=["examples","summary_length","new_article"])
prompt = PromptTemplate(template=template_zero, input_variables=["new_article","summary_length"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

with open("annotated_dataset/data_development.json", "r") as eval_file:
    development_data = json.load(eval_file)

target_article = development_data[3]
example_articles = [x for x in development_data if x['id'] != target_article['id']]

output = []
for article in example_articles:
    new_article=target_article['article']
    summary_length=target_article['summary_length']
    # example = get_example_string_sum_ev(article)
    # summary = llm_chain({"examples": example, "new_article": new_article, "summary_length": summary_length},return_only_outputs=True)
    summary = llm_chain({"new_article": new_article, "summary_length": summary_length},return_only_outputs=True)
    
    references = []
    for key in target_article['rank']:
        if key['rank'] == 1:
            references.append({"Model":key['model'].lower(),
                               "Summary": target_article[key['model'].lower()]})

    final_summary = extract_summary(summary['text'])
    
    # Compute bert-score for each reference
    bert_scores = {}
    for ref in references:
        bert_scores[ref['Model']] = bertscore.compute(predictions=[final_summary], references=[ref['Summary']], model_type="distilbert-base-uncased")

    article_summary = {
        "id_example": article['id'],
        "id_summarized_art": target_article['id'],
        "summary": final_summary,
        "bert-score": bert_scores
    }

    output.append(article_summary)

# Create new directory annotated_dataset
parent_dir = os.getcwd()
create_directory("result", parent_dir)

with open('result/develop_output.json', 'w') as json_file:
    json.dump(output, json_file, indent=4)