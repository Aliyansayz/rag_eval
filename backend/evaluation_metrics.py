from sentence_transformers import SentenceTransformer
import numpy as np


def context_preciseness_score(context, question, answer):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if isinstance(question, str) and isinstance(answer, str): 
        pass
        to_compare = str(question) + "\n" + str(answer)

    texts = [ context , to_compare ]
    embeddings = model.encode(texts)
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    cosine_similarity = f"context preciseness: {cosine_similarity:.4f}"
     
    print(f"context preciseness: {cosine_similarity:.4f}")
    return cosine_similarity


def answer_relevance_score(question, answer):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [ question , answer ]
    embeddings = model.encode(texts)
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    cosine_similarity = f"context recall: {cosine_similarity:.4f}"

    print(f"context recall: {cosine_similarity:.4f}")
    return cosine_similarity




def context_recall_score(context , ground_truth ):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [ context , ground_truth ]
    embeddings = model.encode(texts)
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    cosine_similarity = f"context recall: {cosine_similarity:.4f}"

    print(f"context recall: {cosine_similarity:.4f}")
    return cosine_similarity


def faithfulness_score (actual_output , desired_ouput ):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [ actual_output , desired_ouput ]
    embeddings = model.encode(texts)
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    cosine_similarity = f"Faithfulness Score: {cosine_similarity:.4f}"

    # print(f"Faithfulness Score: {cosine_similarity:.4f}")
    return cosine_similarity


def similarity_score (actual_output , desired_ouput ):

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [ actual_output , desired_ouput ]
    embeddings = model.encode(texts)
    cosine_similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

    print(f"Cosine Similarity: {cosine_similarity:.4f}")
    return cosine_similarity


















def context_precision(context, question, answer):
    context_preciseness_critique_prompt = """
    You will be given a context and a combination of question and answer.
    Your task is to provide a 'total rating' scoring how well context is to the given combination of question and answer.
    Give your answer on a scale of 1 to 5, where 1 means that the context is not precise at all to the question and answer, and 5 means that the context is clearly and unambiguously precising with the context.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating, as a text)
    Total rating: (your rating, as a number between 1 and 5)

    You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

    Now here are the question and context.

    Question: {question}, Answer : {answer}\n
    Context: {context}\n
    Answer::: """

    return context_preciseness_critique_prompt

def question_groundedness(question, context):

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


def answer_relevance(question, answer):
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

    Answer::: """.format(question, answer)

    return answer_relevance_critique_prompt


def faithfulness(answer, ground_truth):
    
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


def context_recall(context , ground_truth):

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


