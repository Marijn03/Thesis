# Abstractive text summarization for high school students using Large Language Models on SCITLDR
## An evaluation of open-source models on computer science papers

In this repository you can find the code I have used to make summaries with the open-source model Llama 3. 
In the thesis, we use the abstract, introduction and conclusion from the SciTLDR training dataset (Cachola et al., 2020).
Their paper can be found here: https://arxiv.org/abs/2004.15011 

In order to run our code, install the packages by the following code:
```
pip install -r requirements.txt
```
We used habrok, a High Performance Computing cluster of the University of Groningen to run Llama 3. This can be done by using the bash file `llama3.sh`.
The `article_txt` directory contains the articles used in the thesis, named by their id. The `article_txt/dataset_prompt` folder consists of .txt files with the article text and the two prompts. In the `models` directory includes the summaries of the closed source models. The `evaluation` directory contains files with the evaluations of the closed-source summaries and the `annotated_dataset` includes our annotated development dataset and the evaluation dataset. Finally, the `results` directory consists of all the table in the paper, the summaries of Llama3 and the evaluation of these summaries. 
