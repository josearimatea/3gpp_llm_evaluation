import requests
import json
import os

def call_llama(user_question, model="tgi", temperature=0.7, max_tokens=5000, stream=False):
    url = "https://llama.venturus.org.br/v1/chat/completions"
    
    # Retrieve the token from the environment variable
    venturus_access_token = os.getenv("VENTURUS_API_KEY")
    if venturus_access_token is None:
        raise ValueError("VENTURUS_API_KEY not found. Please set the environment variable.")
    
    headers = {
        "cf-access-token": venturus_access_token,
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an assistant to answer questions about 3GPP"},
            {"role": "user", "content": user_question}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    
    response = requests.post(url, headers=headers, json=data, stream=stream)
    
    # if response.status_code != 200:
    #     raise Exception(f"Erro na requisição: {response.status_code}, {response.text}")
    
    try:
        answer = json.loads(response.text)['choices'][0]['message']['content']
        return answer
    except (KeyError, IndexError) as e:
        raise Exception(f"Erro ao processar a resposta da API: {e}, {response.text}")
