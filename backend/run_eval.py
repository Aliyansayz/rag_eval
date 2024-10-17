import os, json 
from llm_response import get_response_qwen
from evaluation_metrics import context_precision, context_recall, answer_relevance, faithfulness 
from evaluation_metrics import context_preciseness_score, answer_relevance_score, context_recall_score, faithfulness_score

import pandas as pd


def llm_based_evaluation(question, context, answer, groundtruth): 

    faithfulness_prompt = faithfulness(answer, groundtruth)
    faithfulness_crtique = get_response_qwen(faithfulness_prompt)

    answer_relevance_prompt = answer_relevance(question, answer)
    answer_relevance_crtique = get_response_qwen(answer_relevance_prompt)
    
    context_recall_prompt = context_recall(context , groundtruth)
    context_recall_crtique = get_response_qwen(context_recall_prompt)
    
    context_precision_prompt = context_precision(context, question, answer)
    context_precision_crtique = get_response_qwen(context_precision_prompt)
    
    return faithfulness_crtique, answer_relevance_crtique, context_recall_crtique, context_precision_crtique


def similarity_score_evaluations(question, context, answer, groundtruth): 
    
    f_score = faithfulness_score (answer , groundtruth )
    c_score = context_preciseness_score(context, question, answer)
    ans_score =  answer_relevance_score(question, answer)
    c_recall_score =  context_recall_score(context , groundtruth )
    
    return f_score, c_score, ans_score, c_recall_score


target_file = os.path.abspath('data/tydiqa_id.json')


with open(target_file, 'r') as f:
    data = json.load(f)


rows = []
for i in range(len(data)):
  
  context = data[i].get('context', '')
  query   = data[i].get('input', '')
  groundtruth = data[i].get('expected_output', '')
  answer = get_response_qwen(query)
  f_score, c_score, ans_score, c_recall_score = similarity_score_evaluations(query, context, answer, groundtruth)
  faithfulness_crtique, answer_relevance_crtique, context_recall_crtique, context_precision_crtique = llm_based_evaluation(query, context, answer, groundtruth)


  rows.append([ f_score, c_score, ans_score, c_recall_score,
     faithfulness_crtique, answer_relevance_crtique,
     context_recall_crtique, context_precision_crtique
  ])

df = pd.DataFrame(rows, columns=[ 'faithfulness_score', 'context_preciseness', 'answer_relevance',
                                 'context_recall_score'
                                'faithfulness', 'answer_relevance', 
                                 'context_recall', 'context_precision'])

df.to_csv('context_data.csv', index=False)
