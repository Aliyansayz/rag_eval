import numpy as np

class Rag_Eval:

 
  def get_model(self):
      from sentence_transformers import SentenceTransformer
  
      model = SentenceTransformer('all-MiniLM-L6-v2')
      return model
  
  def get_integer_only(self, text):

      pattern = r"Total rating:\s*(\d+)"
      # Perform the search
      match = re.search(pattern, text)
      # if not match:
      #     return np.nan
      return float(match.group(1))

  
  def context_preciseness_score(self, context, question, answer):
  
      # model = SentenceTransformer('all-MiniLM-L6-v2')
  
      if isinstance(question, str) and isinstance(answer, str):
          pass
          to_compare = str(question) + "\n" + str(answer)
  
      texts = [ context , to_compare ]
      embeddings = model.encode(texts)
      cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
      cosine_similarity = f"{cosine_similarity:.4f}"
  
      # print(f"context preciseness: {cosine_similarity:.4f}")
      return cosine_similarity
  
  
  def answer_relevance_score(self, question, answer):
  
      # model = SentenceTransformer('all-MiniLM-L6-v2')
      texts = [ question , answer ]
      embeddings = model.encode(texts)
      cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
      cosine_similarity = f"{cosine_similarity:.4f}"
  
      # print(f"context recall: {cosine_similarity:.4f}")
      return cosine_similarity
  
  
  
  
  def context_recall_score(self, context , ground_truth ):
  
      # model = SentenceTransformer('all-MiniLM-L6-v2')
      texts = [ context , ground_truth ]
      embeddings = model.encode(texts)
      cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
      cosine_similarity = f"{cosine_similarity:.4f}"
  
      # print(f"context recall: {cosine_similarity:.4f}")
      return cosine_similarity
  
  
  def faithfulness_score (self, actual_output , desired_ouput ):
  
      # model = SentenceTransformer('all-MiniLM-L6-v2')
      texts = [ actual_output , desired_ouput ]
      embeddings = model.encode(texts)
      cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
      cosine_similarity = f"{cosine_similarity:.4f}"
  
      # print(f"Faithfulness Score: {cosine_similarity:.4f}")
      return cosine_similarity
  
  
  def similarity_score (self, actual_output , desired_ouput ):
  
      # model = SentenceTransformer('all-MiniLM-L6-v2')
      texts = [ actual_output , desired_ouput ]
      embeddings = model.encode(texts)
      cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
  
      # print(f"Cosine Similarity: {cosine_similarity:.4f}")
      return cosine_similarity
  
  
  
  
  
  
  def context_precision(self,  context, question, answer):
      context_preciseness_critique_prompt = """
      You will be given a context and a combination of question and answer.
      Your task is to provide a 'total rating' scoring how well context is to the given combination of question and answer.
      Give your answer on a scale of 1 to 5, where 1 means that the context is not precise at all to the question and answer, and 5 means that the context is clearly and unambiguously precising with the context.
  
      Provide your answer as follows:
  
      Answer:::
      
      Total rating: (your rating, as a number between 1 and 5)
  
      You must provide only 'Total rating:' in your answer.
  
      Now here are the question and context.
  
      Question: {question}, Answer : {answer}\n
      Context: {context}\n
      Return the Answer only in integer::: """
  
      return context_preciseness_critique_prompt
  
  def question_groundedness(self,  question, context):
  
      question_groundedness_critique_prompt = """
      You will be given a context and a question.
      Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
      Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.
  
      Provide your answer as follows:
  
      Answer:::
      Evaluation: (your rationale for the rating, as a text)
      Total rating: (your rating, as a number between 1 and 5)
  
      You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
  
      Now here are the question and context.
  
      Question: {question}\n
      Context: {context}\n
      Answer::: """.format(question, context)
  
      return question_groundedness_critique_prompt
  
  
  def answer_relevance(self, question, answer):
      answer_relevance_critique_prompt = """
      You will be given a question and answer.
      Your task is to provide a 'total rating' representing how useful this answer can be to asked question.
      Give your answer on a scale of 1 to 5, where 1 means that the answer is not relevant at all, and 5 means that the answer is extremely useful.
  
      Provide your answer as follows:
  
      Answer:::
      Evaluation: (your rationale for the rating, as a text)
      Total rating: (your rating, as a number between 1 and 5)
  
      You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
  
      Now here is the question and and answer.
  
      Question: {question}\n
      Answer: {answer}\n
  
      Answer::: """.format(question=question, answer=answer)
  
      return answer_relevance_critique_prompt
  
  
  def faithfulness(self, answer, ground_truth):
  
      faithfulness_score_critique_prompt = """
      You will be given a ground truth and an answer.
      Your task is to provide a 'total rating' representing how authentic this answer is to the ground truth.
      Give your answer on a scale of 1 to 5, where 1 means that the answer is not relevant at all, and 5 means that the answer is extremely useful.
  
      Provide your answer as follows:
  
      Answer:::
      Evaluation: (your rationale for the rating, as a text)
      Total rating: (your rating, as a number between 1 and 5)
  
      You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
  
      Now here is the question and and answer.
  
      Ground Truth: {ground_truth} \n
      Answer: {answer} \n
  
      Answer:::
      """.format(answer=answer, ground_truth=ground_truth)
  
      return faithfulness_score_critique_prompt
  
  
  def context_recall(self, context , ground_truth):
  
      context_recall_critique_prompt = """
      You will be given a context and ground_truth.
      Your task is to provide a 'total rating' representing how authentic this context is to the ground truth.
      Give your answer on a scale of 1 to 5, where 1 means that the context is not relevant at all, and 5 means that the context is extremely useful.
  
      Provide your answer as follows:
  
      Answer:::
      Evaluation: (your rationale for the rating, as a text)
      Total rating: (your rating, as a number between 1 and 5)
  
      You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.
  
      Now here is the context and and ground_truth.
  
      Context : {context} \n
      Groundtruth : {ground_truth} \n
  
      Answer:::
  
      """.format(context=context, ground_truth=ground_truth)
  
      return context_recall_critique_prompt



class Eval_Pipeline(Rag_Eval):

  def llm_based_evaluation(self, question, context, answer, groundtruth):

    # get_response_qwen
    llm = LLM_model(model_name)
    
    faithfulness_prompt = faithfulness(answer, groundtruth)
    faithfulness_crtique = llm get_response(faithfulness_prompt) # get_response_llama

    answer_relevance_prompt = answer_relevance(question, answer)
    answer_relevance_crtique = llm get_response(answer_relevance_prompt) # get_response_llama

    context_recall_prompt = context_recall(context , groundtruth)
    context_recall_crtique = llm get_response(context_recall_prompt)

    context_precision_prompt = context_precision(context, question, answer)
    context_precision_crtique = llm get_response(context_precision_prompt)

    return faithfulness_crtique, answer_relevance_crtique, context_recall_crtique, context_precision_crtique


def similarity_score_evaluations(self, question, context, answer, groundtruth):

    f_score = faithfulness_score (answer , groundtruth )
    c_score = context_preciseness_score(context, question, answer)
    ans_score =  answer_relevance_score(question, answer)
    c_recall_score =  context_recall_score(context , groundtruth )

    return f_score, c_score, ans_score, c_recall_score

  
resource = self()

"""
From a structured table/document/csv/json Or Live Rag Response Passing prompt+context to LLM to get rag based Answer. 

  context = data[i].get('context', '')
  query   = data[i].get('input', '')
  groundtruth = data[i].get('expected_output', '')
  # answer 
    ### Change this line
  answer = data[i].get('rag_output', '') # -------------
  rag_answer = answer
"""

f_score, c_score, ans_score, c_recall_score = resource.similarity_score_evaluations(query, context, answer, groundtruth)

faithfulness_crtique, answer_relevance_crtique, context_recall_crtique, context_precision_crtique = resource.llm_based_evaluation(query, context, answer, groundtruth)

faithfulness_crtique_rating = resource.find_total_rating(faithfulness_crtique)
answer_relevance_rating = resource.find_total_rating(answer_relevance_crtique)
context_recall_rating = resource.find_total_rating(context_recall_crtique) 
context_precision_rating = resource.find_total_rating(context_precision_crtique)

rows.append([query, context, rag_answer, groundtruth, f_score, c_score, ans_score, c_recall_score,
     faithfulness_crtique_rating, answer_relevance_rating,
     context_recall_rating, context_precision_rating
  ])

df = pd.DataFrame(rows, columns=['query', 'context', 'rag_answer', 'ground_truth', 'faithfulness_score', 'context_preciseness', 'answer_relevance',
                       'context_recall_score', 'faithfulness', 'answer_relevance',
                                 'context_recall', 'context_precision'])
df.to_csv('rag_report.csv', index=False)
