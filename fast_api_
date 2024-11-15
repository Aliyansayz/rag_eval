from fastapi import FastAPI
from typing import List
from deepeval.metrics.ragas import RAGASAnswerRelevancyMetric
from deepeval.metrics.ragas import RAGASFaithfulnessMetric
from deepeval.metrics.ragas import RAGASContextualRecallMetric
from deepeval.metrics.ragas import RAGASContextualPrecisionMetric






app = FastAPI()


def get_data():
  
    # Replace this with the actual output from your LLM application
  actual_output = "We offer a 30-day full refund at no extra cost."
  
  # Replace this with the expected output from your RAG generator
  expected_output = "You are eligible for a 30 day full refund at no extra cost."
  
  # Replace this with the actual retrieved context from your RAG pipeline
  retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

  return actual_output, expected_output, retrieval_context


# User-defined function to calculate the Faithfulness
def faithful_eval_func():
    metric = RAGASFaithfulnessMetric(threshold=0.5, model="gpt-3.5-turbo")
    test_case = LLMTestCase(
      input="What if these shoes don't fit?",
      actual_output=actual_output,
      expected_output=expected_output,
      retrieval_context=retrieval_context
    )
    return test_case


def answer_relevancy_eval_func():
    metric = RAGASAnswerRelevancyMetric(threshold=0.5, model="gpt-3.5-turbo")
    test_case = LLMTestCase(
      input="What if these shoes don't fit?",
      actual_output=actual_output,
      expected_output=expected_output,
      retrieval_context=retrieval_context
    )
    return test_case


def contextual_recall_eval():
    metric = RAGASContextualRecallMetric(threshold=0.5, model="gpt-3.5-turbo")
    test_case = LLMTestCase(
      input="What if these shoes don't fit?",
      actual_output=actual_output,
      expected_output=expected_output,
      retrieval_context=retrieval_context
    )
    return test_case


def contextual_precision_eval():
    metric = RAGASContextualPrecisionMetric(threshold=0.5, model="gpt-3.5-turbo")
    test_case = LLMTestCase(
      input="What if these shoes don't fit?",
      actual_output=actual_output,
      expected_output=expected_output,
      retrieval_context=retrieval_context
    )
    return test_case




@app.get("/get_example_data")
def get_example_data(numbers: List[float]):
    actual_output, expected_output, retrieval_context = get_data()
    return {"actual_output": actual_output, "expected_output": expected_output, "retrieval_context": retrieval_context}



# FastAPI endpoint to call the function
@app.get("/faithful_eval")
def faithful_eval():
    actual_output, expected_output, retrieval_context = get_data()
    
    test_case = calculate_average()
    metric.measure(test_case)
    print(metric.score)
    return {"numbers": metric.score }


@app.get("/answer_relevancy_eval")
def faithful_eval():
    actual_output, expected_output, retrieval_context = get_data()
    
    test_case = answer_relevancy_eval_func()
    metric.measure(test_case)
    print(metric.score)
    return {"numbers": metric.score }



@app.get("/contextual_recall_eval")
def faithful_eval():
    actual_output, expected_output, retrieval_context = get_data()
    
    test_case = answer_relevancy_eval_func()
    metric.measure(test_case)
    print(metric.score)
    return {"numbers": metric.score }


@app.get("/contextual_precision_eval")
def faithful_eval():
    actual_output, expected_output, retrieval_context = get_data()
    
    test_case = contextual_precision_eval()
    metric.measure(test_case)
    print(metric.score)
    return {"numbers": metric.score }
