import requests
import json


def get_response_llama(prompt:str):

    
    url = "http://34.132.176.211:8000/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "max_tokens": 512 }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json().get("choices")[0].get("message").get("content")

    print(response)

    return response


def get_response_qwen(prompt:str):


    url = "http://34.68.80.87:8000/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "max_tokens": 512 }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json().get("choices")[0].get("message").get("content")

    print(response)

    return response



